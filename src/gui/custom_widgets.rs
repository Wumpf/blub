use egui::*;

pub fn plot_histogram(ui: &mut egui::Ui, height: f32, values: &[f32], top_value: f32, value_unit: &'static str) -> egui::Response {
    let size = vec2(ui.available_size_before_wrap_finite().x, height);
    let (rect, response) = ui.allocate_at_least(size, Sense::hover());
    let style = ui.style().noninteractive();

    let mut shapes = vec![Shape::Rect {
        rect,
        corner_radius: style.corner_radius,
        fill: ui.visuals().extreme_bg_color,
        stroke: ui.style().noninteractive().bg_stroke,
    }];

    let rect = rect.shrink(4.0);
    // tooltip.
    if let Some(pointer_pos) = ui.input().pointer.tooltip_pos() {
        if rect.contains(pointer_pos) {
            let color = ui.visuals().text_color();
            let line_stroke = Stroke::new(1.0, color);

            let y = pointer_pos.y;
            shapes.push(Shape::line_segment([pos2(rect.left(), y), pos2(rect.right(), y)], line_stroke));
            let value = remap(y, rect.bottom_up_range(), 0.0..=top_value);
            let text = format!("{:.1} {}", value, value_unit);
            shapes.push(Shape::text(
                ui.fonts(),
                pos2(rect.left(), y),
                egui::Align2::LEFT_BOTTOM,
                text,
                TextStyle::Monospace,
                color,
            ));
        }
    }

    let line_stroke = Stroke::new(1.0, ui.visuals().weak_text_color());
    let circle_color = ui.visuals().strong_text_color();
    let radius = 2.0;

    for (i, &value) in values.iter().rev().enumerate() {
        let x = remap(i as f32, values.len() as f32..=0.0, rect.x_range());
        let y = remap_clamp(value, 0.0..=top_value, rect.bottom_up_range());

        shapes.push(Shape::line_segment([pos2(x, rect.bottom()), pos2(x, y)], line_stroke));
        if value < top_value {
            shapes.push(Shape::circle_filled(pos2(x, y), radius, circle_color));
        }
    }

    ui.painter().extend(shapes);

    response
}
