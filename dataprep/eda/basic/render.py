"""
This module is for the correct rendering of bokeh plot figures.
"""
# pylint: disable=R0903
# pylint: disable=R0912
from typing import List, Optional

from bokeh.models import Panel, Tabs
from bokeh.plotting import gridplot, show
from dask import compute, delayed

from ..intermediate import Intermediate
from .viz_multi import MultiViz
from .viz_uni import UniViz


class Render:
    # pylint: disable=too-many-instance-attributes
    """
    Encapsulate Renderer functions.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        plot_height_small: int = 300,
        plot_width_small: int = 324,
        plot_height_large: int = 400,
        plot_width_large: int = 450,
        plot_width_wide: int = 972,
        ncolumns: int = 3,
        band_width: float = 1.5,
        tile_size: Optional[float] = None,
        bars: int = 10,
        yscale: str = "linear",
        ascending: bool = False,
    ) -> None:
        self.viz_uni = UniViz()
        self.viz_multi = MultiViz()
        self.plot_height_small = (
            plot_height_small  # set the height of individual plots in the grid.
        )
        self.plot_width_small = (
            plot_width_small  # set the width of individual plots in the grid.
        )
        self.plot_height_large = plot_height_large  # height for large plots
        self.plot_width_large = plot_width_large  # width for large plots
        self.plot_width_wide = plot_width_wide  # wide width for plots
        self.total_cols = (
            ncolumns  # set the total number of columns to be displaced in the grid.
        )
        self.band_width = band_width  # set the band width for the kde plot.
        self.tile_size = tile_size  # set the tile size for the hexbin plot.
        self.bars = bars  # set the max number of bars to show for bar chart.
        self.yscale = yscale  # scale of the y axis labels for the histogram
        self.ascending = ascending  # sort the bars in a bar chart ascending

    def vizualise(
        self, intermediates_list: List[Intermediate], only_x: bool = False
    ) -> None:
        # pylint: disable=too-many-statements
        """
            Shows up the viz on a notebook or browser
        :param intermediates_list: as returned from plot function
        :return: None
        """
        plots = list()
        for intermediate in intermediates_list:
            raw_data = intermediate["raw_data"]
            data_dict = intermediate["result"]

            col_x: str = str()
            col_y: str = str()
            if "col_x" in raw_data:
                col_x = raw_data["col_x"]
                if "col_y" in raw_data:
                    col_y = raw_data["col_y"]

            if col_y is None:
                if "histogram" in data_dict:
                    fig = delayed(self.viz_uni.hist_viz)(
                        data_dict["histogram"],
                        data_dict["missing"],
                        data_dict["orig_df_len"],
                        data_dict["show_y_label"],
                        col_x,
                        self.yscale,
                    )
                    plots.append(fig)
                elif "box_plot" in data_dict:
                    fig = delayed(self.viz_uni.box_viz)(data_dict["box_plot"], col_x)
                    plots.append(fig)
                elif "qqnorm_plot" in data_dict:
                    fig = delayed(self.viz_uni.qqnorm_viz)(
                        data_dict["qqnorm_plot"], col_x
                    )
                    plots.append(fig)
                elif "bar_chart" in data_dict:
                    fig = delayed(self.viz_uni.bar_viz)(
                        data_dict["bar_chart"],
                        data_dict["missing"],
                        col_x,
                        self.bars,
                        self.ascending,
                    )
                    plots.append(fig)
                elif "pie_chart" in data_dict:
                    fig = delayed(self.viz_uni.pie_viz)(
                        data_dict["pie_chart"], col_x, self.bars, self.ascending
                    )
                    plots.append(fig)
                elif "kde_plot" in data_dict:
                    fig = delayed(self.viz_uni.hist_kde_viz)(
                        data_dict["kde_plot"], self.band_width, col_x
                    )
                    plots.append(fig)
            else:
                if "line_chart" in data_dict:
                    fig = delayed(self.viz_multi.line_viz)(
                        data_dict["line_chart"],
                        col_x,
                        col_y,
                        data_dict["grp_cnt_stats"],
                    )
                    plots.append(fig)
                if "nested_bar_chart" in data_dict:
                    fig = delayed(
                        self.viz_multi.nested_viz(
                            data_dict["nested_bar_chart"],
                            col_x,
                            col_y,
                            data_dict["grp_cnt_stats"],
                        )
                    )
                    plots.append(fig)
                elif "stacked_bar_chart" in data_dict:
                    fig = delayed(
                        self.viz_multi.stacked_viz(
                            data_dict["stacked_bar_chart"],
                            data_dict["sub_categories"],
                            col_x,
                            col_y,
                            data_dict["grp_cnt_stats"],
                        )
                    )
                    plots.append(fig)
                elif "heat_map" in data_dict:
                    fig = delayed(
                        self.viz_multi.heat_map_viz(
                            data_dict["heat_map"],
                            col_x,
                            col_y,
                            data_dict["grp_cnt_stats"],
                        )
                    )
                    plots.append(fig)
                elif "scatter_plot" in data_dict:
                    fig = delayed(
                        self.viz_multi.scatter_viz(
                            data_dict["scatter_plot"], col_x, col_y
                        )
                    )
                    plots.append(fig)
                elif "hexbin_plot" in data_dict:
                    fig = delayed(
                        self.viz_multi.hexbin_viz(
                            data_dict["hexbin_plot"],
                            col_x,
                            col_y,
                            tile_size=self.tile_size,
                        )
                    )
                    plots.append(fig)
                elif "box_plot" in data_dict:
                    fig = delayed(self.viz_uni.box_viz)(
                        data_dict["box_plot"],
                        col_x,
                        col_y,
                        grp_cnt_stats=data_dict["grp_cnt_stats"],
                    )
                    plots.append(fig)

        (plots_list,) = compute(plots)

        if only_x:
            tab_list = list()
            for interm, plot in zip(intermediates_list, plots_list):
                plot.height = self.plot_height_large
                plot.width = self.plot_width_large
                if (
                    list(interm["result"].keys())[0] == "stacked_bar_chart"
                    or list(interm["result"].keys())[0] == "nested_bar_chart"
                    or list(interm["result"].keys())[0] == "heat_map"
                ):
                    plot.height = self.plot_height_small
                    plot.width = self.plot_width_wide
                tab = Panel(child=plot, title=list(interm["result"].keys())[0])
                tab_list.append(tab)
            show(Tabs(tabs=tab_list))
        else:
            show(
                gridplot(
                    children=plots_list,
                    sizing_mode=None,
                    toolbar_location=None,
                    ncols=self.total_cols,
                    plot_height=self.plot_height_small,
                    plot_width=self.plot_width_small,
                )
            )