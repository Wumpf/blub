use futures::{Future, FutureExt};
use std::{ops::Range, pin::Pin};

pub struct GpuTimerScopeResult {
    pub label: String,
    /// Time range of this scope in seconds.
    /// Meaning of absolute value is not defined.
    pub time: Range<f64>,

    pub nested_scopes: Vec<GpuTimerScopeResult>,
}

#[derive(Default)]
struct QueryPoolQueryAddress {
    pool_idx: u32,
    query_idx: u32,
}

#[derive(Default)]
struct UnprocessedTimerScope {
    label: String,
    start_query: QueryPoolQueryAddress,
    nested_scopes: Vec<UnprocessedTimerScope>,
}

struct QueryPool {
    query_set: wgpu::QuerySet,

    buffer: wgpu::Buffer,
    buffer_mapping: Option<Pin<Box<dyn Future<Output = std::result::Result<(), wgpu::BufferAsyncError>>>>>,

    capacity: u32,
    num_used_queries: u32,
    num_resolved_queries: u32,
}

impl QueryPool {
    const MIN_CAPACITY: u32 = 64;

    fn new(capacity: u32, device: &wgpu::Device) -> Self {
        info!("Creating new GpuProfiler QueryPool with {} elements", capacity);
        QueryPool {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                ty: wgpu::QueryType::Timestamp,
                count: capacity,
            }),

            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuProfiler - Query Buffer"),
                size: (wgpu::QUERY_SIZE * capacity) as u64,
                usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
                mapped_at_creation: false,
            }),
            buffer_mapping: None,

            capacity,
            num_used_queries: 0,
            num_resolved_queries: 0,
        }
    }

    fn reset(&mut self) {
        self.num_used_queries = 0;
        self.num_resolved_queries = 0;
        self.buffer_mapping = None;
        self.buffer.unmap();
    }

    fn resolved_buffer_slice(&self) -> wgpu::BufferSlice {
        self.buffer.slice(0..(self.num_resolved_queries * wgpu::QUERY_SIZE) as u64)
    }
}

#[derive(Default)]
struct PendingFrame {
    query_pools: Vec<QueryPool>,
    closed_scopes: Vec<UnprocessedTimerScope>,
}

pub struct GpuProfiler {
    pub enable_timer: bool,
    pub enable_debug_marker: bool,

    unused_pools: Vec<QueryPool>,

    pending_frames: Vec<PendingFrame>,
    active_frame: PendingFrame,
    open_scopes: Vec<UnprocessedTimerScope>,

    max_num_pending_frames: usize,
    timestamp_to_sec: f64,
}

impl GpuProfiler {
    /// max_num_pending_frames: How many pending profiler-frames can be in flight at a time.
    /// A profiler-frame is in flight until its queries have been successfully resolved with process_finished_queries.
    /// If this threshold is reached, end_frame will drop frames. (typical values are 2~4)
    ///
    /// timestamp_period: Result of wgpu::Queue::get_timestamp_period()
    ///
    pub fn new(max_num_pending_frames: usize, timestamp_period: f32) -> Self {
        assert!(max_num_pending_frames > 0);
        GpuProfiler {
            enable_timer: true,
            enable_debug_marker: true,

            unused_pools: Vec::new(),

            pending_frames: Vec::new(),
            active_frame: PendingFrame {
                query_pools: Vec::new(),
                closed_scopes: Vec::new(),
            },
            open_scopes: Vec::new(),

            max_num_pending_frames,
            timestamp_to_sec: timestamp_period as f64 / 1000.0 / 1000.0 / 1000.0,
        }
    }

    // Reserves two query objects.
    // Our query pools always have an even number of queries, so we know the next query is the next in the same pool.
    fn allocate_query_pair(&mut self, device: &wgpu::Device) -> QueryPoolQueryAddress {
        let num_pools = self.active_frame.query_pools.len();

        if let Some(active_pool) = self.active_frame.query_pools.last_mut() {
            if active_pool.capacity > active_pool.num_used_queries {
                let address = QueryPoolQueryAddress {
                    pool_idx: num_pools as u32 - 1,
                    query_idx: active_pool.num_used_queries,
                };
                active_pool.num_used_queries += 2;
                assert!(active_pool.num_used_queries <= active_pool.capacity);
                return address;
            }
        }

        let new_pool = if let Some(reused_pool) = self.unused_pools.pop() {
            reused_pool
        } else {
            QueryPool::new(
                self.active_frame
                    .query_pools
                    .iter()
                    .map(|pool| pool.capacity)
                    .sum::<u32>()
                    .max(QueryPool::MIN_CAPACITY),
                device,
            )
        };
        self.active_frame.query_pools.push(new_pool);

        QueryPoolQueryAddress {
            pool_idx: self.active_frame.query_pools.len() as u32 - 1,
            query_idx: 0,
        }
    }

    pub fn begin_scope<Recorder: ProfilerCommandRecorder>(&mut self, label: &str, encoder_or_pass: &mut Recorder, device: &wgpu::Device) {
        if self.enable_timer {
            let start_query = self.allocate_query_pair(device);

            encoder_or_pass.write_timestamp(
                &self.active_frame.query_pools[start_query.pool_idx as usize].query_set,
                start_query.query_idx,
            );

            self.open_scopes.push(UnprocessedTimerScope {
                label: String::from(label),
                start_query: start_query,
                ..Default::default()
            });
        }
        if self.enable_debug_marker {
            encoder_or_pass.push_debug_group(label);
        }
    }

    pub fn end_scope<Recorder: ProfilerCommandRecorder>(&mut self, encoder_or_pass: &mut Recorder) {
        if self.enable_timer {
            let open_scope = self.open_scopes.pop().expect("No profiler GpuProfiler scope was previously opened");
            encoder_or_pass.write_timestamp(
                &self.active_frame.query_pools[open_scope.start_query.pool_idx as usize].query_set,
                open_scope.start_query.query_idx + 1,
            );
            if let Some(open_parent_scope) = self.open_scopes.last_mut() {
                open_parent_scope.nested_scopes.push(open_scope);
            } else {
                self.active_frame.closed_scopes.push(open_scope);
            }
        }
        if self.enable_debug_marker {
            encoder_or_pass.pop_debug_group();
        }
    }

    /// Puts query resolve commands in the encoder for all unresolved, pending queries of the current profiler frame.
    pub fn resolve_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
        for query_pool in self.active_frame.query_pools.iter_mut() {
            if query_pool.num_resolved_queries == query_pool.num_used_queries {
                continue;
            }
            assert!(query_pool.num_resolved_queries < query_pool.num_used_queries);
            encoder.resolve_query_set(
                &query_pool.query_set,
                query_pool.num_resolved_queries..query_pool.num_used_queries,
                &query_pool.buffer,
                (query_pool.num_resolved_queries * wgpu::QUERY_SIZE) as u64,
            );
            query_pool.num_resolved_queries = query_pool.num_used_queries;
        }
    }

    fn cache_unused_query_pools(&mut self, mut query_pools: Vec<QueryPool>) {
        // TODO: Drop query pools that are clearly too small
        for pool in query_pools.iter_mut() {
            pool.reset();
        }
        self.unused_pools.append(&mut query_pools);
    }

    /// Marks the end of a frame.
    pub fn end_frame(&mut self) -> Result<(), ()> {
        // TODO: Error messages
        if !self.open_scopes.is_empty() {
            return Err(());
        }
        if self
            .active_frame
            .query_pools
            .iter()
            .any(|pool| pool.num_resolved_queries != pool.num_used_queries)
        {
            return Err(());
        }

        // Make sure we don't overflow
        if self.pending_frames.len() == self.max_num_pending_frames {
            // Drop previous frame.
            let dropped_frame = self.pending_frames.pop().unwrap();
            self.cache_unused_query_pools(dropped_frame.query_pools);
            // TODO report this somehow
        }

        // Map all buffers.
        for pool in self.active_frame.query_pools.iter_mut() {
            pool.buffer_mapping = Some(pool.resolved_buffer_slice().map_async(wgpu::MapMode::Read).boxed());
        }

        // Enqueue
        let mut frame = Default::default();
        std::mem::swap(&mut frame, &mut self.active_frame);
        self.pending_frames.push(frame);

        assert!(self.pending_frames.len() <= self.max_num_pending_frames);

        Ok(())
    }

    fn process_timings_recursive(
        timestamp_to_sec: f64,
        resolved_query_buffers: &Vec<wgpu::BufferView>,
        unprocessed_scopes: Vec<UnprocessedTimerScope>,
    ) -> Vec<GpuTimerScopeResult> {
        unprocessed_scopes
            .into_iter()
            .map(|scope| {
                let nested_scopes = if scope.nested_scopes.is_empty() {
                    Vec::new()
                } else {
                    Self::process_timings_recursive(timestamp_to_sec, resolved_query_buffers, scope.nested_scopes)
                };

                // By design timestamps for start/end are consecutive.
                let buffer_offset = (scope.start_query.query_idx * wgpu::QUERY_SIZE) as usize;
                let raw_timestamps: &[u64; 2] = bytemuck::from_bytes(
                    &resolved_query_buffers[scope.start_query.pool_idx as usize][buffer_offset..(buffer_offset + std::mem::size_of::<u64>() * 2)],
                );

                GpuTimerScopeResult {
                    label: scope.label,
                    time: (raw_timestamps[0] as f64 * timestamp_to_sec)..(raw_timestamps[1] as f64 * timestamp_to_sec),
                    nested_scopes,
                }
            })
            .collect()
    }

    /// Checks if all timer queries for the oldest pending finished frame are done and returns that snapshot if any.
    pub fn process_finished_queries(&mut self) -> Option<Vec<GpuTimerScopeResult>> {
        let frame = self.pending_frames.first_mut()?;

        // We only process if all mappings succeed.
        if frame
            .query_pools
            .iter_mut()
            .any(|pool| (&mut pool.buffer_mapping.as_mut().unwrap()).now_or_never().is_none())
        {
            return None;
        }

        let frame = self.pending_frames.remove(0);

        let results = {
            let resolved_query_buffers: Vec<wgpu::BufferView> = frame
                .query_pools
                .iter()
                .map(|pool| pool.resolved_buffer_slice().get_mapped_range())
                .collect();
            Self::process_timings_recursive(self.timestamp_to_sec, &resolved_query_buffers, frame.closed_scopes)
        };

        self.cache_unused_query_pools(frame.query_pools);

        Some(results)
    }
}

pub trait ProfilerCommandRecorder {
    fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32);
    fn push_debug_group(&mut self, label: &str);
    fn pop_debug_group(&mut self);
}

macro_rules! ImplProfilerCommandRecorder {
    ($($name:ident $(< $lt:lifetime >)?,)*) => {
        $(
            impl $(< $lt >)? ProfilerCommandRecorder for wgpu::$name $(< $lt >)? {
                fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, query_index: u32) {
                    self.write_timestamp(query_set, query_index)
                }

                fn push_debug_group(&mut self, label: &str) {
                    self.push_debug_group(label)
                }

                fn pop_debug_group(&mut self) {
                    self.pop_debug_group()
                }
            }
        )*
    };
}

ImplProfilerCommandRecorder!(CommandEncoder, RenderPass<'a>, ComputePass<'a>,);
