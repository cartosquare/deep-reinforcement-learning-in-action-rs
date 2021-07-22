use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;

// plot scatter graph using xy data and save to disk
// path: destination file
// xy: x-axe and y-axe data
// xmin: x-axe min value
// xmax: x-axe max value
// ymin: y-axe min value
// ymax: y-axe max value
// xlabel: x-axe label
// ylabel: y-axe label
pub fn xy_scatter_plot(
	path: String,
	xy: Vec<(f64, f64)>,
	xmin: f64,
	xmax: f64,
	ymin: f64,
	ymax: f64,
	xlabel: String,
	ylabel: String,
) {
	let plot: Plot = Plot::new(xy).point_style(
		PointStyle::new()
			.marker(PointMarker::Circle)
			.colour("#DD3355")
			.size(1.0),
	);

	let view = ContinuousView::new()
		.add(plot)
		.x_range(xmin, xmax)
		.y_range(ymin, ymax)
		.x_label(xlabel)
		.y_label(ylabel);

	Page::single(&view).save(path).unwrap();
}
