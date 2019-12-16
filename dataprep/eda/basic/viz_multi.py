"""
This module implements functions for plotting visualizations for two fields.
"""
# pytype: disable=import-error
import math
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    FactorRange,
    HoverTool,
    LinearColorMapper,
    PrintfTickFormatter,
    BasicTicker,
    Legend,
    FuncTickFormatter,
)
from bokeh.palettes import viridis, Pastel1  # pylint: disable=E0611
from bokeh.plotting import figure
from bokeh.transform import linear_cmap, transform
from ..utils import sample_n
from ..palette import PALETTE, BIPALETTE

TOOLS = ""


class MultiViz:
    """
    Encapsulation for multivariate vizualisations.
    """

    nested_cat: bool = False
    stacked_cat: bool = False
    heatmap_cat: bool = False
    scatter: bool = False
    hexbin: bool = False
    max_label_len: int = 15  # maximum length of an axis label

    @classmethod
    def _make_title(cls, grp_cnt_stats: Dict[str, int], col_x: str, col_y: str) -> Any:
        """
        Make title of plot
        :param grp_cnt_stats: group count statistics
        :param col_x: column X
        :param col_y: column Y
        :return: title
        """

        x_total = grp_cnt_stats["x_total"]
        x_show = grp_cnt_stats["x_show"]
        if "y_total" in grp_cnt_stats:
            y_total = grp_cnt_stats["y_total"]
            y_show = grp_cnt_stats["y_show"]

            if x_total > x_show and y_total > y_show:
                return "(top {} out of {}) {} by (top {} out of {}) {}".format(
                    y_show, y_total, col_y, x_show, x_total, col_x
                )
            if y_total > y_show:
                return "(top {} out of {}) {} by {}".format(
                    y_show, y_total, col_y, col_x
                )
        if x_total > x_show:
            return "{} by (top {} out of {}) {}".format(col_y, x_show, x_total, col_x)
        return "{} by {}".format(col_y, col_x)

    def nested_viz(
        self,
        data: Dict[Tuple[Any, Any], int],
        col_x: str,
        col_y: str,
        grp_cnt_stats: Dict[str, int],
    ) -> Any:
        """
        Nested categories plot
        :param data: the intermediates result
        :param col_x: column X
        :param col_y: column Y
        :param grp_cnt_stats: group count statistics
        :return: Bokeh Plot Figure
        """
        x_values = list([tuple(map(str, i)) for i in data.keys()])
        counts = list(data.values())
        data_source = ColumnDataSource(data=dict(x_values=x_values, counts=counts))
        title = self._make_title(grp_cnt_stats, col_x, col_y)

        plot_figure = figure(
            x_range=FactorRange(*x_values),
            tools=TOOLS,
            toolbar_location=None,
            title=title,
        )

        plot_figure.vbar(
            x="x_values",
            top="counts",
            width=1,
            source=data_source,
            line_color="white",
            line_width=3,
        )
        plot_figure.add_tools(
            HoverTool(
                tooltips=[("Category", "@x_values"), ("Count", "@counts")], mode="mouse"
            )
        )
        plot_figure.y_range.start = 0
        plot_figure.x_range.range_padding = 0.03
        plot_figure.xgrid.grid_line_color = None
        plot_figure.yaxis.axis_label = "Count"
        plot_figure.xaxis.major_label_orientation = math.pi / 2
        plot_figure.title.text_font_size = "10pt"
        plot_figure.xaxis.formatter = FuncTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (self.max_label_len, self.max_label_len)
        )
        self.nested_cat = True
        return plot_figure

    def stacked_viz(  # pylint: disable=too-many-arguments
        self,
        data: Dict[str, int],
        sub_categories: List[str],
        col_x: str,
        col_y: str,
        grp_cnt_stats: Dict[str, int],
    ) -> Any:
        """
        Stacked categories plot
        :param data: the intermediates result
        :param sub_categories: list of all subcategories
        :param col_x: column X
        :param col_y: column Y
        :param grp_cnt_stats: group count statistics
        :return: Bokeh Plot Figure
        """
        title = self._make_title(grp_cnt_stats, col_x, col_y)
        plot_figure = figure(
            x_range=data["x_categories"],
            toolbar_location=None,
            title=title,
            tools="hover",
            tooltips=[
                ("Category", "@x_categories, $name"),
                ("Percentage", "@$name{0.2f}%"),
            ],
        )
        palette = Pastel1[9] * (len(sub_categories) // len(Pastel1) + 1)
        if "Others" in sub_categories:
            colours = palette[0 : len(sub_categories) - 1] + ["#636363"]
        else:
            colours = palette[0 : len(sub_categories)]

        renderers = plot_figure.vbar_stack(
            stackers=sub_categories,
            x="x_categories",
            width=0.9,
            source=data,
            line_width=1,
            color=colours,
        )

        legend_it = [(cat, [rend]) for cat, rend in zip(sub_categories, renderers)]
        legend = Legend(items=legend_it)
        legend.label_text_font_size = "8pt"
        plot_figure.add_layout(legend, "right")

        plot_figure.y_range.start = 0
        plot_figure.x_range.range_padding = 0.03
        plot_figure.xgrid.grid_line_color = None
        plot_figure.yaxis.axis_label = "Percent"
        plot_figure.xaxis.major_label_orientation = math.pi / 3
        plot_figure.title.text_font_size = "10pt"
        plot_figure.xaxis.formatter = FuncTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (self.max_label_len, self.max_label_len)
        )
        self.stacked_cat = True
        return plot_figure

    def heat_map_viz(
        self, data: pd.DataFrame, col_x: str, col_y: str, grp_cnt_stats: Dict[str, int]
    ) -> Any:
        """
        Stacked categories plot
        :param data: the intermediates result
        :param col_x: column X
        :param col_y: column Y
        :param grp_cnt_stats: group count statistics
        :return: Bokeh Plot Figure
        """
        title = self._make_title(grp_cnt_stats, col_x, col_y)

        source = ColumnDataSource(data)
        palette = BIPALETTE[(len(BIPALETTE) // 2 - 1) :]
        mapper = LinearColorMapper(
            palette=palette, low=data.total.min() - 0.01, high=data.total.max()
        )
        x_cats = list(set(data[col_x]))
        y_cats = list(set(data[col_y]))
        plot_figure = figure(
            x_range=x_cats,
            y_range=y_cats,
            toolbar_location=None,
            tools=TOOLS,
            x_axis_location="below",
            title=title,
        )

        renderer = plot_figure.rect(
            x=col_x,
            y=col_y,
            width=1,
            height=1,
            source=source,
            line_color=None,
            fill_color=transform("total", mapper),
        )

        color_bar = ColorBar(
            color_mapper=mapper,
            location=(0, 0),
            ticker=BasicTicker(desired_num_ticks=7),
            formatter=PrintfTickFormatter(format="%d"),
        )
        plot_figure.add_tools(
            HoverTool(
                tooltips=[
                    (col_x, "@{}".format(col_x)),
                    (col_y, "@{}".format(col_y)),
                    ("Count", "@total"),
                ],
                mode="mouse",
                renderers=[renderer],
            )
        )
        plot_figure.add_layout(color_bar, "right")
        plot_figure.xaxis.major_label_orientation = math.pi / 3
        plot_figure.xgrid.grid_line_color = None
        plot_figure.ygrid.grid_line_color = None
        plot_figure.xaxis.formatter = FuncTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (self.max_label_len, self.max_label_len)
        )
        plot_figure.yaxis.formatter = FuncTickFormatter(
            code="""
            if (tick.length > %d) return tick.substring(0, %d-2) + '...';
            else return tick;
        """
            % (self.max_label_len, self.max_label_len)
        )
        self.heatmap_cat = True
        return plot_figure

    def scatter_viz(  # pylint: disable=C0330, R0914
        self,
        points: List[Tuple[Any, Any]],
        col_x: str,
        col_y: str,
        sample_size: int = 1000,
    ) -> Any:
        """
        Scatter plot
        :param points: list of points to be plotted
        :param col_x: column X
        :param col_y: column Y
        :param sample_size: sample size for points to plot
        :return: Bokeh Plot Figure
        """
        x_values = np.array([t[0] for t in points])
        y_values = np.array([t[1] for t in points])

        x_values, y_values = (
            sample_n(x_values, sample_size),
            sample_n(y_values, sample_size),
        )

        if len(x_values) == sample_size:
            title = "{} by {} (sample size {})".format(col_y, col_x, sample_size)
        else:
            title = "{} by {}".format(col_y, col_x)

        plot_figure = figure(tools=TOOLS, title=title, toolbar_location=None)

        renderer = plot_figure.circle(
            x_values, y_values, color=PALETTE[0], size=4, name="points"
        )

        plot_figure.add_tools(
            HoverTool(
                renderers=[renderer],
                tooltips=[("x", "@x"), ("y", "@y")],
                mode="mouse",
                names=["points"],
            )
        )

        plot_figure.xaxis.axis_label = "{}".format(col_x)
        plot_figure.yaxis.axis_label = "{}".format(col_y)
        plot_figure.xaxis.major_label_orientation = math.pi / 4
        self.scatter = True
        plot_figure.title.text_font_size = "10pt"
        return plot_figure

    def hexbin_viz(  # pylint: disable=C0330, R0914, R0913
        self,
        points: List[Tuple[Any, Any]],
        col_x: str,
        col_y: str,
        sample_size: int = 1000,
        tile_size: Optional[float] = None,
    ) -> Any:
        """
        Modified hexbin scatter plot
        :param points: list of points to be plotted
        :param col_x: column X
        :param col_y: column Y
        :param sample_size: sample size for points to plot
        :param tile_size: hex tile size
        :return: Bokeh Plot Figure
        """
        x_values = np.array([t[0] for t in points])
        y_values = np.array([t[1] for t in points])
        x_values, y_values = (
            sample_n(x_values, sample_size),
            sample_n(y_values, sample_size),
        )

        if tile_size is None:
            xmin, xmax = np.nanmin(x_values), np.nanmax(x_values)
            xsize = (xmax - xmin) // 20
            ymin, ymax = np.nanmin(y_values), np.nanmax(y_values)
            ysize = (ymax - ymin) // 20

            tile_size = max(xsize, ysize) + 1

        if len(x_values) == sample_size:
            title = "{} by {} (sample size {})".format(col_y, col_x, sample_size)
        else:
            title = "{} by {}".format(col_y, col_x)

        plot_figure = figure(
            match_aspect=True,
            tools=TOOLS,
            title=title,
            toolbar_location=None,
            background_fill_color="#f5f5f5",
        )
        plot_figure.grid.visible = False

        cmap = list(reversed(viridis(256)))

        renderer, bins = plot_figure.hexbin(
            x_values,
            y_values,
            size=tile_size,
            hover_alpha=0.8,
            palette=cmap,
            hover_color="pink",
            line_color="white",
            aspect_scale=1,
        )

        if bins.counts.size == 0:
            max_bin_count = 0
            min_bin_count = 0
        else:
            max_bin_count = max(bins.counts)
            min_bin_count = min(bins.counts)

        color_mapper = linear_cmap("c", cmap, min_bin_count, max_bin_count)

        plot_figure.add_tools(
            HoverTool(
                tooltips=[("Count", "@c")],
                point_policy="follow_mouse",
                renderers=[renderer],
            )
        )

        color_bar = ColorBar(
            color_mapper=color_mapper["transform"], width=8, location=(0, 0)
        )
        plot_figure.add_layout(color_bar, "right")

        plot_figure.xaxis.axis_label = "{}".format(col_x)
        plot_figure.yaxis.axis_label = "{}".format(col_y)
        plot_figure.xaxis.major_label_orientation = math.pi / 4
        self.hexbin = True
        plot_figure.title.text_font_size = "10pt"
        return plot_figure

    def line_viz(  # pylint: disable=C0330, R0914
        self,
        data: Dict[str, Tuple[Any, Any]],
        col_x: str,
        col_y: str,
        grp_cnt_stats: Dict[str, int],
    ) -> Any:
        """
        Multi-line graph
        :param data: the intermediates result
        :param col_x: column X
        :param col_y: column Y
        :param grp_cnt_stats: group count statistics
        :return: Bokeh Plot Figure
        """
        categories = list(data.keys())
        palette = PALETTE * (len(categories) // len(PALETTE) + 1)
        title = self._make_title(grp_cnt_stats, col_x, col_y)

        plot_figure = figure(tools=TOOLS, title=title, toolbar_location=None)

        plot_dict = dict()
        for cat, colour in zip(categories, palette):
            ticks = [
                (data[cat][1][i] + data[cat][1][i + 1]) / 2
                for i in range(len(data[cat][1]) - 1)
            ]
            cat_name = (
                (cat[: (self.max_label_len - 1)] + "...")
                if len(cat) > self.max_label_len
                else cat
            )

            source = ColumnDataSource(
                {
                    "x": ticks,
                    "y": data[cat][0],
                    "left": data[cat][1][:-1],
                    "right": data[cat][1][1:],
                }
            )
            plot_dict[cat_name] = plot_figure.line(
                x="x", y="y", source=source, color=colour
            )
            plot_figure.add_tools(
                HoverTool(
                    renderers=[plot_dict[cat_name]],
                    tooltips=[
                        ("{}".format(col_x), "{}".format(cat)),
                        ("Frequency", "@y"),
                        ("{} bin".format(col_y), "[@left, @right]"),
                    ],
                    mode="mouse",
                )
            )
        legend = Legend(items=[(x, [plot_dict[x]]) for x in plot_dict])
        plot_figure.add_layout(legend, "right")
        plot_figure.yaxis.axis_label = "Frequency"
        plot_figure.xaxis.axis_label = col_y

        return plot_figure