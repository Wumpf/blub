use egui::*;

pub fn plot_barchart(
    ui: &mut egui::Ui,
    size: Vec2,
    values: &[f32],
    top_value: f32,
    value_unit: &'static str,
    value_decimals: usize,
) -> egui::Response {
    let (rect, response) = ui.allocate_at_least(size, Sense::hover());
    let style = ui.style().noninteractive();

    let mut shapes = vec![Shape::Rect {
        rect,
        corner_radius: style.corner_radius,
        fill: ui.visuals().extreme_bg_color,
        stroke: ui.style().noninteractive().bg_stroke,
    }];

    let rect = rect.shrink(4.0);
    let half_bar_width = rect.width() / values.len() as f32 * 0.5;

    for (i, &value) in values.iter().rev().enumerate() {
        let x = remap(i as f32, values.len() as f32..=0.0, rect.x_range());
        let x_min = ui.painter().round_to_pixel(x - half_bar_width);
        let x_max = ui.painter().round_to_pixel(x + half_bar_width);
        let y = remap_clamp(value, 0.0..=top_value, rect.bottom_up_range());
        let bar = Rect {
            min: pos2(x_min, y),
            max: pos2(x_max, rect.bottom()),
        };

        let mut fill_color = ui.visuals().weak_text_color();

        let tooltip = if let Some(pointer_pos) = ui.input().pointer.interact_pos() {
            if bar.contains(pointer_pos) {
                fill_color = ui.visuals().text_color();
                Some(Shape::text(
                    ui.fonts(),
                    pointer_pos,
                    egui::Align2::LEFT_BOTTOM,
                    format!("{:.*} {}", value_decimals, value, value_unit),
                    TextStyle::Body,
                    ui.visuals().strong_text_color(),
                ))
            } else {
                None
            }
        } else {
            None
        };

        shapes.push(Shape::Rect {
            rect: bar,
            corner_radius: 0.0,
            fill: fill_color,
            stroke: Default::default(),
        });

        // tooltip.
        if let Some(tooltip) = tooltip {
            shapes.push(tooltip);
        }
    }

    ui.painter().extend(shapes);

    response
}
