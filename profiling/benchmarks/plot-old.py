from lib import Benchmark


class PlotCorrelationV1(Benchmark):
    def bench(self) -> None:
        from dataprep.eda.correlation_old import plot_correlation
        from tempfile import TemporaryDirectory
        import dask.dataframe as dd
        import pandas as pd

        if self.reader == "dask":
            if self.dpath.suffix == ".parquet":
                df = dd.read_parquet(self.dpath)
            elif self.dpath.suffix == ".csv":
                df = dd.read_csv(self.dpath)
        elif self.reader == "pandas":
            if self.dpath.suffix == ".parquet":
                df = pd.read_parquet(self.dpath)
            elif self.dpath.suffix == ".csv":
                df = pd.read_csv(self.dpath)
        else:
            raise NotImplementedError(f"Reader {self.reader} not implemented")

        with TemporaryDirectory() as tdir:
            plot_correlation(df).save(f"{tdir}/report.html")


if __name__ == "__main__":
    PlotCorrelationV1().run()
