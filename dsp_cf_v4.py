#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import re
import subprocess

print(f"Python version used by batch {sys.version}")


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"Package {package} installed successfully...")


def uninstall(package):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package, "--yes"])
    print(f"Package {package} uninstalled successfully...")


def upgrade(package):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--upgrade"]
    )
    print(f"Package {package} upgraded successfully...")


# install("loguru==0.7.2")
# install("joblib")
# install("awswrangler")
# install("xgboost")
# install("gluonts")
# install("shap")
# install("mxnet==1.6.0")
# install("numpy==1.23.1")
# uninstall("typing_extensions")
# upgrade("typing_extensions==4.7.1")
# upgrade("pandas==1.5.3")


import pandas as pd
import numpy as np
import boto3
import json
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from multiprocessing import Pool
from functools import partial
import shap
from uuid import uuid4
import io
from io import StringIO, BytesIO

from loguru import logger
import joblib

import psycopg2

# import rs_utility as utl
import awswrangler as wr

from sklearn.linear_model import LinearRegression, ElasticNet, QuantileRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

from xgboost import XGBRegressor
import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer
from gluonts.mx.model import deepar
from gluonts.evaluation import make_evaluation_predictions, Evaluator

np.random.seed(6)
os.environ["PYTHONHASHSEED"] = str(6)


# removed_features = ['attr_rate_y0_2']


class DataHandler(object):
    def __init__(self, bucket):
        self.bucket = bucket

    def get_conn(self, cluster_name):
        """
        The function is used to run queries on AMZL Analytics Redshift Cluster using RJDBC driver

        Params:
        -----------------------
        cluster_name = redshift cluster name (choose from amzlanalytics, amzl-bia-compute)

        Returns:
        -----------------------
        connection = connection object to run queries on the Redshift Cluster

        """

        secret_name = "redshift-jinye-access"
        region_name = "us-east-1"

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=region_name)
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response["SecretString"]
        secret_dict = json.loads(secret)
        User = list(secret_dict.keys())[0]
        Password = secret_dict[User]

        if cluster_name == "amzlanalytics":
            Host = "amzlanalytics.cfc3pypfgclf.us-east-1.redshift.amazonaws.com"
            Dbname = "amzlanalytics"

        elif cluster_name == "amzl-bia-compute":
            Host = "amzl-bia-compute.cfc3pypfgclf.us-east-1.redshift.amazonaws.com"
            Dbname = "amzlbiacompute"

        Port = 8192
        connection = psycopg2.connect(
            database=Dbname, user=User, password=Password, port=Port, host=Host
        )
        return connection

    def get_df(self, query: str, credential: str):
        """
        The function is used to return the pandas DataFrame post run of Query on Redshift Cluster

        Params:
        -----------------
        query = SQL query that needs to run on the Redshift Cluster
        credential = Redshift Cluster Name

        Returns:
        -----------------
        df = pandas DataFrame after run of query

        """

        with self.get_conn(credential) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result_set = cur.fetchall()
                colnames = [desc.name for desc in cur.description]
                df = pd.DataFrame.from_records(result_set, columns=colnames)
        return df

    def read_csv_from_s3(self, s3_path):
        """
        This function is used to read the csv files from s3 bucket

        Params:
        -----------------
        s3_path = s3 prefix, all file names in the folder will be extracted as file_names

        Returns:
        -----------------
        None

        """
        s3_client = boto3.client("s3")
        obj = s3_client.get_object(Bucket=self.bucket, Key=s3_path)
        df = pd.read_csv(obj["Body"])
        return df

    def read_parquet_from_s3(self, s3_path, **args):
        """
        Load parquet file from s3 into a dataframe

        """
        s3_client = boto3.client("s3")
        obj = s3_client.get_object(Bucket=self.bucket, Key=s3_path)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()), **args)
    
    def write_parquet_to_s3(self, df, s3_path):
        """
        transform a dataframe into a parquet file and store it in s3
        """
        out_buffer = BytesIO()
        df.to_parquet(out_buffer, index=False)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(self.bucket, s3_path).put(Body=out_buffer.getvalue())
        logger.debug("file upload to S3 completed")
        
    def read_pickle_from_s3(self, s3_path):
        """
        Load pickle file from s3
        """
        s3 = boto3.resource('s3')
        pickle_file = joblib.load(s3.Bucket(self.bucket).Object(s3_path).get()['Body'].read())
        return pickle_file
        
    def write_pickle_to_s3(self, df, s3_path):
        """
        transform a file into a pickle file and store it in s3
        """
        out_buffer = BytesIO()
        joblib.dump(df, out_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, s3_path).put(Body=out_buffer.getvalue())
        logger.debug("file upload to S3 completed")

    def read_s3_keys_from_prefix(self, s3_path):
        """
        This function is used to read the csv file names from s3 bucket

        Params:
        -----------------
        s3_path = s3 prefix, all file names in the folder will be extracted as file_names

        Returns:
        -----------------
        None

        """
        s3_client = boto3.client("s3")
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=s3_path)

        s3_keys = []
        for page in pages:
            for obj in page["Contents"]:
                s3_keys.append(obj["Key"])
        return s3_keys

    def download_from_s3(self, s3_download_path, local_download_path):
        """
        The function is used to download files from s3 bucket to local directory using boto3 s3 resource client

        Params:
        -----------------
        bucket: string
                s3 bucket used to download and upload objects
        download_object_path: string
                object prefix in s3 buckets for download action
        download_local_path: string
                object path in local directory for storing in local action

        Returns:
        -----------------
        None

        """
        s3_client = boto3.client("s3")
        try:
            s3_client.download_file(self.bucket, s3_download_path, local_download_path)
            logger.debug(
                f"Downloaded file from s3 bucket {self.bucket} and saved to {self.local_path}"
            )
        except:
            logger.error(f"File missing in s3 location")

    def upload_to_s3(self, local_download_path, s3_upload_path):
        """
        The function is used to upload files to s3 bucket

        Params:
        -------------------------
        self.bucket: s3 bucket name
        local_download_path: local file path that needs to be uploaded to s3
        s3_upload_path: object path in s3 bucket, contains prefix and file name

        Returns:
        -------------------------
        None

        """
        s3 = boto3.client("s3")
        try:
            response = s3.upload_file(local_download_path, self.bucket, s3_upload_path)
            logger.debug(f"File {local_download_path} Uploaded to s3 successfully")
        except ClientError as e:
            logger.error(e)

    def upload_file_from_codecommit_to_s3(
        self, cc_client, repository, file_path, s3_client, bucket_name, key=""
    ):
        """
        This function is used to push the latest code changes to s3 bucket to update batch job script

        Params:
        -------------------------
        cc_client: code commit boto3 client
        repository: code commit repository name which stores the code files
        s3_client: s3 boto3 client
        bucket_name: name of the bucket to upload the code file
        key: s3_path prefix for above bucket name

        Returns:
        -------------------------
        True: if function is able to run successfully

        """
        repo_file = cc_client.get_file(repositoryName=repository, filePath=file_path)

        basename = file_path.split("/")[-1]
        temp_fname = f"{str(uuid4())}-{basename}"
        content = repo_file["fileContent"]
        decoded = content
        decoded = str(decoded.decode("utf-8"))
        regex = r"""\\n(?=(?:[^'"]|'[^']*'|"[^"]*")*$)"""
        decoded = [str(x) for x in re.split(regex, decoded)]
        with open(temp_fname, "w") as f:
            f.writelines(x for x in decoded)

        if key == "":
            key = file_path.replace("\\", "/")
        with open(temp_fname, "rb") as f:
            s3_client.upload_fileobj(f, bucket_name, key)
        os.remove(temp_fname)
        return True


class FeatureHandler(DataHandler):
    def __init__(
        self,
        input_data_path,
        train_stage_config_dict,
        performance_stage_config_dict,
        prediction_length,
        s3_object_path_dict,
        bucket=None,
    ):
        self.train_stage_config_dict = train_stage_config_dict
        self.performance_stage_config_dict = performance_stage_config_dict
        self.stage3_config = performance_stage_config_dict["stage3_config"]
        self.s3_object_path_dict = s3_object_path_dict
        self.input_data_path = input_data_path
        self.prediction_length = prediction_length
        self.trainer = Trainer(
            ctx=mx.cpu(),
            epochs=5,
            num_batches_per_epoch=500,
            learning_rate=2e-3,
            clip_gradient=10.0,
            weight_decay=1e-4,
            hybridize=False,
        )
        self.evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        DataHandler.__init__(self, bucket)

    def generate_univariate_forecasts(
        self,
        prediction_length,
        input_file_path,
        model_local_path,
        s3_univariate_model_upload_path,
        predictions_local_file_path,
        s3_predictions_upload_path,
        s3_upload_flag,
        forecast_type="Train",
    ):
        """
        This function is used to generate univariate time series forecast features for training process.

        Params:
        ---------------------
        prediction_length: int
                length of prediction period
        model_local_path: str
                local path for storing training model pickle file
        s3_univariate_model_upload_path: str
                s3 path for storing training model pickle file

        Returns:
        ---------------------
        df_forecast: pd.DataFrame
                dataframe with univariate time series forecast features at dsp-station level

        """
        if forecast_type == "Train":
            run_week = (datetime.today() - timedelta(365)).isocalendar()[1]
        else:
            run_week = (datetime.today()).isocalendar()[1]
        prediction_week = run_week + prediction_length
        prediction_length = prediction_length
        freq = "1W"

        train_data = pd.read_parquet(input_file_path)
        train_data = train_data[
            [
                "station_pair",
                "dsp_code",
                "year",
                "week",
                "year_week",
                "completed_routes_max",
            ]
        ]
        train_data = train_data.loc[train_data["year_week"] <= run_week, :]
        train_data = train_data.astype(
            {"completed_routes_max": "float64", "year": "int64", "week": "int64"}
        )
        train_data = train_data.sort_values(
            by=["station_pair", "year", "week"]
        ).reset_index(drop=True)

        df_station_pair = (
            train_data.groupby(["station_pair"])
            .agg({"completed_routes_max": "count", "year": "max", "week": "max"})
            .reset_index()
            .sort_values(by=["stage_start"], ascending=False)
            .reset_index(drop=True)
        )
        df_station_pair = df_station_pair.loc[
            df_station_pair.completed_routes_max > 2.5 * prediction_length, :
        ]

        gluonts_df = train_data[
            train_data.station_pair.isin(df_station_pair.station_pair.unique())
        ]
        gluonts_df = gluonts_df[["station_pair", "stage_start", "completed_routes_max"]]

        df_dict = []
        for station_pair, df in gluonts_df.groupby(["station_pair"]):
            df["stage_start"] = df.apply(
                lambda x: x["stage_start"]
                - (timedelta((x["stage_start"].weekday() + 1) % 7)),
                axis=1,
            )
            df.set_index(df.stage_start.values, inplace=True)
            date_range = pd.DataFrame(
                pd.date_range(start=df.index.min(), end="2023-01-01", freq="W"),
                columns=["stage_start"],
            )
            df = date_range.merge(df, on=["stage_start"], how="left")
            df["station_pair"].fillna(method="ffill", inplace=True)
            df["completed_routes_max"].fillna(
                df.completed_routes_max.median(), inplace=True
            )
            df.set_index(df.stage_start.values, inplace=True)
            df.drop(columns=["stage_start"], inplace=True)
            df_dict.append(df)

        df_gl = pd.concat(df_dict, axis=0)
        logger.debug(f"Total DSPs in the dataset {df_gl.station_pair.nunique()}")

        train_list = []
        for station_pair, time_series in df_gl.groupby(["station_pair"]):
            train_list.append(
                {
                    "target": time_series.completed_routes_max[:-prediction_length],
                    #                                   "item_id":time_series.station_pair[:-prediction_length],
                    "start": pd.Timestamp(time_series.index.min()),
                }
            )
        train_dataset = ListDataset(train_list, freq=freq)

        val_list = []
        for time_series in df_dict:
            val_list.append(
                {
                    "target": time_series.completed_routes_max,
                    #                           "item_id":time_series.station_pair,
                    "start": time_series.index.min(),
                }
            )
        val_dataset = ListDataset(val_list, freq=freq)

        dar_estimator = deepar.DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=self.trainer,
            use_feat_static_cat=False,
        )

        dar_predictor = dar_estimator.train(training_data=train_dataset)

        with open(model_local_path, "wb") as f:
            joblib.dump(dar_predictor, f)

        if s3_upload_flag:
            self.upload_to_s3(
                local_download_path=model_local_path,
                s3_upload_path=s3_univariate_model_upload_path,
            )

        dar_forecast_it, ts_it = make_evaluation_predictions(
            dataset=val_dataset,
            predictor=dar_predictor,
            num_samples=100,
        )

        dar_forecasts = list(dar_forecast_it)
        tss = list(ts_it)

        forecast_dict = []
        for i, df in enumerate(df_dict):
            df_temp = df[-prediction_length:].copy()
            df_temp["quantile_forecast_10_y1"] = dar_forecasts[i].quantile(0.1)
            df_temp["quantile_forecast_50_y1"] = dar_forecasts[i].quantile(0.5)
            df_temp["quantile_forecast_90_y1"] = dar_forecasts[i].quantile(0.9)
            forecast_dict.append(df_temp)

        df_forecasts = pd.concat(forecast_dict, axis=0)
        df_forecasts["week"] = (df_forecasts.index + timedelta(1)).isocalendar().week
        df_forecasts.to_csv(predictions_local_file_path)
        if s3_upload_flag:
            self.upload_to_s3(
                local_download_path=predictions_local_file_path,
                s3_upload_path=s3_predictions_upload_path,
            )

        logger.debug(
            f"Generating univariate time series forecasts at station pair level completed."
        )

    def merge_univariate_forecasts(self, univariate_forecast_local_file_path, df_train):
        """
        This function is used to merge univariate forecasts to training data

        """

        df_forecasts = pd.read_csv(univariate_forecast_local_file_path)
        df_forecasts = df_forecasts[df_forecasts.stage == self.stage3_config[0]][
            [
                "station_pair",
                "quantile_forecast_10_y1",
                "quantile_forecast_50_y1",
                "quantile_forecast_90_y1",
            ]
        ]

        df_train = df_train.merge(df_forecasts, on="station_pair", how="left")
        df_station = (
            df_train.groupby(["station_code"])[
                [
                    "quantile_forecast_10_y1",
                    "quantile_forecast_50_y1",
                    "quantile_forecast_90_y1",
                ]
            ]
            .median()
            .reset_index()
        )

        df_train = df_train.merge(
            df_station,
            on=["station_code"],
            how="left",
            suffixes=["_station_pair", "_station"],
        )

        df_train["quantile_forecast_10_y1_station_pair"].fillna(
            df_train["quantile_forecast_10_y1_station"], inplace=True
        )
        df_train["quantile_forecast_50_y1_station_pair"].fillna(
            df_train["quantile_forecast_50_y1_station"], inplace=True
        )
        df_train["quantile_forecast_90_y1_station_pair"].fillna(
            df_train["quantile_forecast_90_y1_station"], inplace=True
        )
        return df_train

    def impute_using_fallback_logic(self, df):
        """
        This function is used to fill missing values for features using fallback to previous stages

        """
        # filling missing values using fall back logic - separate function for fall back logics
        # future to be update using univariate forecasts fallback  logic #Jin_0822: please check if this has been corrected
        df["ratio_station_3_5"].fillna(df["ratio_region_3_5"], inplace=True)

        df["completed_routes_0"].fillna(df["completed_routes_2"], inplace=True)
        df["completed_routes_0"].fillna(df["completed_routes_1"], inplace=True)
        df["completed_routes_2"].fillna(df["completed_routes_1"], inplace=True)
        df["completed_routes_3"].fillna(df["completed_routes_4"], inplace=True)
        df["completed_routes_3"].fillna(df["completed_routes_5"], inplace=True)
        df["completed_routes_4"].fillna(df["completed_routes_5"], inplace=True)
        # df["completed_routes_max_0"].fillna(df["completed_routes_max_2"], inplace=True)
        df["completed_routes_max_0"].fillna(df["completed_routes_max_1"], inplace=True)
        # df["completed_routes_max_2"].fillna(df["completed_routes_max_1"], inplace=True)
        df["completed_routes_max_3"].fillna(df["completed_routes_max_4"], inplace=True)
        df["completed_routes_max_3"].fillna(df["completed_routes_max_5"], inplace=True)
        df["completed_routes_max_4"].fillna(df["completed_routes_max_5"], inplace=True)
        # df["requested_routes_max_0"].fillna(df["requested_routes_max_2"], inplace=True)
        df["requested_routes_max_0"].fillna(df["requested_routes_max_1"], inplace=True)
        # df["requested_routes_max_2"].fillna(df["requested_routes_max_1"], inplace=True)
        df["requested_routes_max_3"].fillna(df["requested_routes_max_4"], inplace=True)
        df["requested_routes_max_3"].fillna(df["requested_routes_max_5"], inplace=True)
        df["requested_routes_max_4"].fillna(df["requested_routes_max_5"], inplace=True)
        # df["final_route_target_2"].fillna(df["final_route_target_1"], inplace=True)
        df["t6_cr_isight_points_1"].fillna(0, inplace=True)
        df["t6_isight_points_1"].fillna(0, inplace=True)

        logger.debug("Missing Values Imputation using Fallback logic is completed")
        return df

    def do_feature_engineering(self, df, feature_list, dataset_type="Train"):
        """
        This function is used to perform feature engineering on both train and inference datasets

        """
        df_pivot = pd.pivot_table(
            df,
            values=feature_list,
            index=["station_pair", "station_code", "dsp_code", "region"],
            columns="stage",
            aggfunc="median",
        )
        df_pivot.columns = [
            "_".join([str(c) for c in c_list]) for c_list in df_pivot.columns.values
        ]
        df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1).reset_index()

        df_pivot_max = pd.pivot_table(
            df,
            values=["completed_routes_max", "requested_routes_max"],
            index=["station_pair"],
            columns="stage",
            aggfunc="max",
        )
        df_pivot_max.columns = [
            "_max_".join([str(c) for c in c_list])
            for c_list in df_pivot_max.columns.values
        ]
        df_pivot_max = df_pivot_max.reindex(
            sorted(df_pivot_max.columns), axis=1
        ).reset_index()

        df_pivot = df_pivot.merge(df_pivot_max, on=["station_pair"], how="left")

        most_recent_dsp_type = (
            df.groupby(["station_pair", "station_code", "dsp_code"])
            .agg({"dsp_type": "last"})["dsp_type"]
            .reset_index()
        )

        df_pivot = df_pivot.merge(
            most_recent_dsp_type,
            on=["station_pair", "station_code", "dsp_code"],
            how="left",
        )

        # Calculate the station scaling factors for prior year
        station_3_5 = (
            df_pivot[
                [
                    "station_code",
                    "completed_routes_3",
                    "completed_routes_5",
                ]
            ]
            .dropna()
            .groupby(["station_code"])[["completed_routes_3", "completed_routes_5"]]
            .sum()
            .reset_index()
        )

        station_3_5["ratio"] = (
            station_3_5.completed_routes_3 / station_3_5.completed_routes_5
        )

        station_3_5.drop(
            columns=["completed_routes_3", "completed_routes_5"], inplace=True
        )

        # Calculate the region scaling factors for prior year
        region_3_5 = (
            df_pivot[["region", "completed_routes_3", "completed_routes_5"]]
            .dropna()
            .groupby(["region"])[["completed_routes_3", "completed_routes_5"]]
            .sum()
            .reset_index()
        )

        region_3_5["ratio"] = (
            region_3_5["completed_routes_3"] / region_3_5["completed_routes_5"]
        )

        region_3_5.drop(
            columns=["completed_routes_3", "completed_routes_5"], inplace=True
        )

        df = df_pivot.merge(station_3_5, on=["station_code"], how="left").merge(
            region_3_5, on=["region"], suffixes=["_station_3_5", "_region_3_5"]
        )
        
        df.loc[df.completed_routes_4 == 0, "completed_routes_4"] = np.nan
        df["completed_routes_4"].fillna(df["completed_routes_5"], inplace=True)
        df["completed_routes_4"].fillna(df["completed_routes_4"].mean(), inplace=True)

        # ===== calculate scaling factors between different stages at station pair level =======
        df["ratio_station_pair_3_5"] = (
            df["completed_routes_3"] / df["completed_routes_5"]
        )
        df["ratio_station_pair_3_4"] = (
            df["completed_routes_3"] / df["completed_routes_4"]
        )
        # ===== calculate scaling factors between different stages at station pair level =======

        df["station_pair_3_5_route_diff"] = df["completed_routes_3"] * (
            1 - 1 / df["ratio_station_pair_3_5"]
        )

        df["station_pair_1_5_route_diff"] = (
            df["completed_routes_1"] - df["completed_routes_5"]
        ).fillna(df["completed_routes_3"] / df["ratio_station_pair_3_5"])

        df["region_completed_routes_1"] = df.groupby(df.region)[
            "completed_routes_1"
        ].transform("sum")
        df["region_completed_routes_3"] = df.groupby(df.region)[
            "completed_routes_3"
        ].transform("sum")
        df["region_completed_routes_5"] = df.groupby(df.region)[
            "completed_routes_5"
        ].transform("sum")

        df["region_3_5_routes_diff"] = (
            df["region_completed_routes_3"] - df["region_completed_routes_5"]
        )

        df["station_completed_routes_1"] = df.groupby(df.station_code)[
            "completed_routes_1"
        ].transform("sum")
        df["station_completed_routes_3"] = df.groupby(df.station_code)[
            "completed_routes_3"
        ].transform("sum")
        df["station_completed_routes_5"] = df.groupby(df.station_code)[
            "completed_routes_5"
        ].transform("sum")

        df["station_3_5_routes_diff"] = df["station_completed_routes_3"] * (
            1 - 1 / df["ratio_station_3_5"]
        )

        # fleet feature for last year prediction week
        df["region_vin_branded_3"] = df.groupby(df.region)["vin_branded_3"].transform(
            "sum"
        )
        df["region_vin_total_3"] = df.groupby(df.region)["vin_total_3"].transform("sum")
        df["region_vin_branded_rate_3"] = (
            df["region_vin_branded_3"] / df["region_vin_total_3"]
        )

        # defining prediction week features
        if dataset_type == "Performance":
            df["vin_branded_0"] = df["vin_branded_1"]
            df["vin_total_0"] = df["vin_total_1"]
            df["vin_branded_rate_0"] = df["vin_branded_rate_1"]
            df["vin_active_rate_2"] = df["vin_active_rate_1"]
            df["completed_routes_0"] = np.nan
            df["requested_routes_max_0"] = np.nan
            df["completed_routes_max_0"] = np.nan
            df["incentives_0"] = np.nan

        # forward looking fleet features - to be sourced from fleet team
        df["region_vin_branded_0"] = df.groupby(df.region)["vin_branded_0"].transform(
            "sum"
        )
        df["region_vin_total_0"] = df.groupby(df.region)["vin_total_0"].transform("sum")
        df["region_vin_branded_rate_0"] = (
            df["region_vin_branded_0"] / df["region_vin_total_0"]
        )

        df["ratio_station_pair_to_station_1"] = (
            df["completed_routes_1"] / df["station_completed_routes_1"]
        )  # currently used as model feature

        logger.debug("Feature Engineering Completed.")

        return df

    def merge_transfers_traindata(self, run_date, train_input):
        """
        This function is used to merge the transfers DSP type to training data and add distance feature to the model

        """
        train_run_date = run_date - timedelta(365)
        dsp_type_dict = {
            "Recruit to Expand": "Pop-Up",
            "Recruit to Expand - DSPx": "Pinnacle",
            "Recruit to Transfer": "Internal-Transfer",
            "Recruit to Offer": "New-DSP",
        }

        targets = self.read_csv_from_s3(self.s3_object_path_dict["transfers_path"])
        targets.loc[:, "target_launch_date"] = pd.to_datetime(
            targets["target_launch_date"]
        )
        targets = targets[
            (
                targets.target_launch_date
                >= pd.to_datetime(train_run_date - timedelta(168))
            )
            & (
                targets.target_launch_date
                <= pd.to_datetime(
                    train_run_date + timedelta(7 * self.prediction_length)
                )
            )
        ].reset_index(drop=True)
#         targets = targets[targets.cycle != "CYCLE_2"].reset_index(drop=True)

        targets = targets.rename(columns={"station": "station_code"})
        targets.loc[:, "dsp_type"] = targets["recruitment_type"].map(dsp_type_dict)

        if targets.shape[0] < 200:
            raise Exception("Targets Data has less than 200 rows")

        dsp_type_dict_new = dict(zip(targets.station_pair, targets.dsp_type))

        train_input.loc[
            train_input.station_pair.isin(targets.station_pair), "dsp_type"
        ] = (train_input["station_pair"].map(dsp_type_dict_new)).fillna(
            train_input.dsp_type.mode().values[0]
        )

        dsp_distance_dict = dict(zip(targets.station_pair, targets.distance))
        train_input.loc[:, "distance"] = (
            train_input["station_pair"].map(dsp_distance_dict).fillna(0)
        )

        # Turning off features for targets prior to stage 2 config
        train_input.loc[:, "vol_share_cycle1_1"] = np.where(
            ~train_input.dsp_type.isin(["DSP 2.0", "DSP 1.0", "Walker"]),
            0,
            train_input["vol_share_cycle1_1"],
        )
        #         train_input.loc[:, "requested_routes_max_1"] = np.where(
        #             ~train_input.dsp_type.isin(["DSP 2.0", "DSP 1.0", "Walker"]), 0, train_input["requested_routes_max_1"]
        #         )
        #         train_input.loc[:, "completed_routes_max_3"] = np.where(
        #             ~train_input.dsp_type.isin(["DSP 2.0", "DSP 1.0", "Walker"]), 0, train_input["completed_routes_max_3"]
        #         )
        train_input.loc[:, "vin_branded_1"] = np.where(
            ~train_input.dsp_type.isin(["DSP 2.0", "DSP 1.0", "Walker"]),
            0,
            train_input["vin_branded_1"],
        )
        train_input.loc[:, "vin_branded_3"] = np.where(
            ~train_input.dsp_type.isin(["DSP 2.0", "DSP 1.0", "Walker"]),
            0,
            train_input["vin_branded_3"],
        )

        train_input.loc[:, "completed_routes_max_1_tmp"] = train_input[
            "completed_routes_max_1"
        ].fillna(10)

        train_input.loc[:, "completed_routes_max_1"] = np.where(
            train_input.dsp_type == "Internal-Transfer",
            train_input["completed_routes_max_1"],
            train_input["completed_routes_max_1_tmp"],
        )

        train_input.loc[:, "requested_routes_max_1_tmp"] = train_input[
            "requested_routes_max_1"
        ].fillna(0)

        train_input.loc[:, "requested_routes_max_1"] = np.where(
            train_input.dsp_type == "Internal-Transfer",
            train_input["requested_routes_max_1"],
            train_input["requested_routes_max_1_tmp"],
        )
        train_input.loc[:, "completed_routes_max_3_tmp"] = train_input[
            "completed_routes_max_3"
        ].fillna(0)

        train_input.loc[:, "completed_routes_max_3"] = np.where(
            train_input.dsp_type == "Internal-Transfer",
            train_input["completed_routes_max_3"],
            train_input["completed_routes_max_3_tmp"],
        )

        # train_input['completed_routes_max_y0_1_max'].fillna(10, inplace=True)
        # Jin Edit on 9/22
        train_input["completed_routes_2"].fillna(10, inplace=True)
        train_input["final_route_target_2"].fillna(10, inplace=True)

        if train_input.shape[0] < 200:
            raise Exception("Train Input has less than 200 rows")

        return train_input

    def merge_volume_actuals_data(self, train_stage_config_dict, train_input):
        """
        This function is used to merge forward looking volume actuals feature to the training input

        """
        df = self.read_csv_from_s3(
            self.s3_object_path_dict["station_volume_actuals_path"]
        )

        df["week"] = ["%02d" % x for x in df.week]
        df["year_week"] = df["year"].astype(str) + "-" + df["week"].astype(str)
        df = (
            df.groupby(["year_week", "year", "week", "country_code", "station_code"])
            .agg({"volume_actuals": "sum", "work_days": "sum"})
            .reset_index()
        )
        df.loc[:, "lrp_station_vf_0"] = np.round(df["volume_actuals"] / df["work_days"])
        df = df.sort_values(by=["station_code", "year_week"])
        df = df[
            df.year_week.isin(train_stage_config_dict["prediction_year_week"])
        ].reset_index(drop=True)
        df = df.reindex(columns=["station_code", "lrp_station_vf_0"])
        df["lrp_station_vf_0"] = df["lrp_station_vf_0"].fillna(
            df["lrp_station_vf_0"].mean()
        )

        train_input = train_input.merge(df, on=["station_code"], how="left")
        return train_input

    def do_target_upsampling(self, df):
        """
        This function is used to perform upsampling of targets to improve their representation in the training dataset

        """
        df_dsp = df[df.dsp_type == "DSP"].reset_index(drop=True)
        df_target = df[df.dsp_type != "DSP"].reset_index(drop=True)

        upscale_factor = (df.station_pair.nunique() * 0.20) / df_target.shape[0]
        strata_list = [
            "station_pair",
            "station_code",
            "dsp_code",
            "region",
            "dsp_type",
        ]

        df_target["stratified_col"] = df_target[strata_list].apply(
            lambda x: ", ".join(x.astype(str)), axis=1
        )

        df_target_sampled = df_target.groupby("stratified_col", group_keys=False).apply(
            lambda x: x.sample(
                frac=upscale_factor, replace=True, random_state=6, ignore_index=True
            )
        )
        df_target_sampled.drop(columns=["stratified_col"], axis=1, inplace=True)

        df = pd.concat([df_dsp, df_target_sampled], axis=0)
        df = df.reset_index(drop=True)
        logger.debug(
            f"Upsampling completed, total number of targets augmented = {df_target_sampled.shape[0] - df_target.shape[0]}"
        )

        return df

    def create_training_input(
        self,
        train_stage_config_dict,
        feature_list,
        run_date,
        univariate_local_file_path,
        local_download_path,
        s3_upload_path,
        s3_upload_flag,
    ):
        """
        The job of this function is to filter the input data for last year stage 1 and do custom processing steps

        Params:
        -----------------------

        Returns:
        -----------------------
        df_train: Dataframe
                Training Dataset ready for model training

        Example:
        -----------------------
        train_input = create_training_input()

        """
        df_train = self.read_parquet_from_s3(self.input_data_path)
        df_train["year_week"] = (
            df_train["year"].astype(str) + "-" + df_train["week"].astype(str)
        )

        df_train.loc[:, "stage"] = 999
        df_train.loc[
            df_train.year_week.isin(train_stage_config_dict["prediction_year_week"]),
            "stage",
        ] = 0
        df_train.loc[
            df_train.year_week.isin(train_stage_config_dict["stage1_config"]), "stage"
        ] = 1
        df_train.loc[
            df_train.year_week.isin(train_stage_config_dict["stage2_config"]), "stage"
        ] = 2
        df_train.loc[
            df_train.year_week.isin(train_stage_config_dict["stage3_config"]), "stage"
        ] = 3
        df_train.loc[
            df_train.year_week.isin(train_stage_config_dict["stage4_config"]), "stage"
        ] = 4
        df_train.loc[
            df_train.year_week.isin(train_stage_config_dict["stage5_config"]), "stage"
        ] = 5

        df_train = df_train[df_train.stage != 999].reset_index(drop=True)
        df_train = self.do_feature_engineering(df_train, feature_list)
        df_train = self.impute_using_fallback_logic(df_train)
        df_train = self.merge_volume_actuals_data(train_stage_config_dict, df_train)

        df_train.loc[:, "distance"] = 0

        df_train = self.merge_transfers_traindata(run_date, df_train)

        numeric_columns = df_train.select_dtypes(include="number").columns.tolist()
        dsp_columns = numeric_columns + ["dsp_code"]
        dsp_metrics = (
            df_train.reindex(columns=dsp_columns)
            .groupby("dsp_code")
            .median()
            .reset_index()
        )
        dsp_metrics = df_train[["dsp_code"]].merge(
            dsp_metrics, on="dsp_code", how="left"
        )

        # First we replace the missing values using DSP code
        common_cols = list(set(df_train.columns).intersection(set(dsp_metrics.columns)))
        for cols in common_cols:
            df_train.loc[:, cols] = df_train[cols].fillna(dsp_metrics[cols])

        # If DSP code is not available, then use Station code to replace remaining missing values

        station_columns = numeric_columns + ["station_code"]
        station_metrics = (
            df_train.reindex(columns=station_columns)
            .groupby(["station_code"])
            .median()
            .reset_index()
        )
        station_metrics = df_train[["station_code"]].merge(
            station_metrics, on="station_code", how="left"
        )

        common_cols = list(
            set(df_train.columns).intersection(set(station_metrics.columns))
        )
        for cols in common_cols:
            df_train.loc[:, cols] = df_train[cols].fillna(station_metrics[cols])

        # df_train = df_train.fillna(
        #     df_train.median()
        # )  # Jin_0822, Divye add a function for data distribution based similarity imputations P1

        # Upsampling of targets
        # df_train = self.do_target_upsampling(df_train)

        # Divye_0929: add categorical features for targets
        dsp_cat = pd.get_dummies(df_train["dsp_type"], dtype=float)
        dsp_type_list = [
            "DSP 2.0",
            "DSP 1.0",
            "Walker",
            "Internal-Transfer",
            "Pinnacle",
            "Pop-Up",
            "New-DSP",
            "Transfer Outs",
            "Exiting",
        ]
        if len(dsp_cat.columns) < len(dsp_type_list):
            missing_cols = set(dsp_type_list).difference(dsp_cat)
            for cols in missing_cols:
                dsp_cat[cols] = 0

        df_train = pd.concat([df_train, dsp_cat], axis=1)

        df_train["label"] = df_train["completed_routes_max_0"]

        # outlier treatment for targets
        df_train["label"] = np.where(
            (df_train.station_pair_tenure_0 >= 7)
            & (df_train.label < 15)
            & (df_train.dsp_type != "DSP"),
            15,
            df_train["label"],
        )
        df_train["completed_routes_2"] = np.where(
            (df_train.station_pair_tenure_0 >= 7)
            & (df_train.completed_routes_2 < 15)
            & (df_train.dsp_type != "DSP"),
            15,
            df_train["completed_routes_2"],
        )

        # outlier treatment for active da count: New-DSPs, Pop-ups ramping period
        df_train["active_da_count_0"] = np.where(
            (df_train.station_pair_tenure_0 <= 7)
            & (df_train.completed_routes_2 <= 30)
            & (df_train.dsp_type.isin(["Pop-Up", "New-DSP"])),
            30,
            df_train["active_da_count_0"],
        )

        try:
            os.makedirs(
                local_download_path.split("/dsp_peak_scaling_train_input.pqt")[0]
            )
        except:
            logger.debug("Directory already exists")

        # df_train = df_train.loc[df_train.dsp_type=='DSP']  #Jin Edit to test 9/21
        df_train.to_parquet(local_download_path, index=False)

        if s3_upload_flag:
            self.upload_to_s3(
                local_download_path=local_download_path, s3_upload_path=s3_upload_path
            )
        logger.debug(f"Training Input Created with shape {df_train.shape}")

    def stage_performance_input(
        self,
        train_input_path,
        performance_stage_config_dict,
        feature_list,
    ):
        """
        The function is used to create inference data

        Params:
        -----------------------
        training_input: DataFrame
                Input Data for Model Training

        Returns:
        -----------------------
        df_val: Dataframe
                Inference dataset post preprocessing

        Example:
        -----------------------
        valid_input = create_inference_input(training_input)

        """
        df_performance = self.read_parquet_from_s3(self.input_data_path)
        df_performance["year_week"] = (
            df_performance["year"].astype(str)
            + "-"
            + df_performance["week"].astype(str)
        )

        df_performance.loc[:, "stage"] = 999
        df_performance.loc[
            df_performance.year_week.isin(
                performance_stage_config_dict["prediction_year_week"]
            ),
            "stage",
        ] = 0
        df_performance.loc[
            df_performance.year_week.isin(
                performance_stage_config_dict["stage1_config"]
            ),
            "stage",
        ] = 1
        df_performance.loc[
            df_performance.year_week.isin(
                performance_stage_config_dict["stage2_config"]
            ),
            "stage",
        ] = 2
        df_performance.loc[
            df_performance.year_week.isin(
                performance_stage_config_dict["stage3_config"]
            ),
            "stage",
        ] = 3
        df_performance.loc[
            df_performance.year_week.isin(
                performance_stage_config_dict["stage4_config"]
            ),
            "stage",
        ] = 4
        df_performance.loc[
            df_performance.year_week.isin(
                performance_stage_config_dict["stage5_config"]
            ),
            "stage",
        ] = 5

        df_performance = df_performance[df_performance.stage != 999].reset_index(
            drop=True
        )
        df_performance = self.do_feature_engineering(
            df_performance, feature_list, dataset_type="Performance"
        )

        df_performance.loc[:, "distance"] = 0
        # df_performance.loc[:, "dsp_type"] = "DSP"

        df_train = pd.read_parquet(train_input_path)
        df_performance.loc[:, "incentives_0"] = np.nan
        df_performance.loc[:, "vin_branded_rate_0"] = np.nan
        df_performance.loc[:, "vin_active_rate_2"] = np.nan
        df_performance.loc[:, "region_vin_branded_rate_0"] = np.nan

        df_performance = df_performance.merge(
            df_train[
                [
                    "station_pair",
                    "incentives_0",
                    "vin_branded_rate_0",
                    "vin_active_rate_2",
                ]
            ],
            on="station_pair",
            how="left",
            suffixes=["", "_fallback"],
        ).merge(
            df_train[["region", "region_vin_branded_rate_0"]].drop_duplicates(),
            on="region",
            how="left",
            suffixes=["", "_fallback"],
        )

        df_performance["incentives_0"].fillna(
            df_performance["incentives_0_fallback"], inplace=True
        )
        df_performance["vin_branded_rate_0"].fillna(
            df_performance["vin_branded_rate_0_fallback"], inplace=True
        )
        df_performance["vin_active_rate_2"].fillna(
            df_performance["vin_active_rate_2_fallback"], inplace=True
        )
        df_performance["region_vin_branded_rate_0"].fillna(
            df_performance["region_vin_branded_rate_0_fallback"], inplace=True
        )

        df_performance["label"] = 0
        return df_performance

    def create_stage2_data(
        self, performance_stage_config_dict, model_version, run_date
    ):
        """
        This function is used to create mtp, lpt and stp forecast data for adding stage 2 features in feature lab

        """
        # preparing DCPM historical publish data
        try:
            keys = self.read_s3_keys_from_prefix(
                f"inferences_v4/publish_date={run_date - timedelta(7)}"
            )
        except:
            keys = []

        if len(keys) > 0:
            result = []
            for path in keys:
                try:
                    df = self.read_csv_from_s3(path)
                except:
                    df = self.read_parquet_from_s3(path)
                result.append(df)
                dcpm_data = pd.concat(result)
                dcpm_data["prediction_week"] = ["%02d" % x for x in dcpm_data.prediction_week]
                dcpm_data["prediction_year_week"] = dcpm_data.prediction_year.astype(str) + "-" + dcpm_data.prediction_week.astype(str)
                dcpm_data["stage"] = 999
                dcpm_data.loc[
                    dcpm_data["prediction_year_week"].isin(
                        performance_stage_config_dict["stage2_config"]
                    ),
                    "stage",
                ] = 2
                dcpm_data = dcpm_data[dcpm_data.stage != 999].reset_index(drop=True)
                dcpm_data.sort_values(
                    by=["station_pair", "prediction_year", "prediction_week"],
                    inplace=True,
                )
                dcpm_data.rename(
                    columns={
                        "prediction_year_week": "year_week",
                        "predictions_no_incentives": "dcpm_routes_max",
                    },
                    inplace=True,
                )
                dcpm_data = dcpm_data[["station_pair", "year_week", "dcpm_routes_max"]]
        else:
            dcpm_data = pd.DataFrame()

        # preparing STP Data
        stp_data = self.read_csv_from_s3(self.s3_object_path_dict["stp_path"])
        stp_data["year_week"] = (
            stp_data["year"].astype(str) + "-" + stp_data["week"].astype(str)
        )
        stp_data["stage"] = 999
        stp_data.loc[
            stp_data["year_week"].isin(performance_stage_config_dict["stage2_config"]),
            "stage",
        ] = 2
        stp_data.sort_values(by=["station_pair", "year", "week"], inplace=True)
        stp_data = stp_data[stp_data.stage != 999].reset_index(drop=True)
        stp_data = stp_data[["station_pair", "year_week", "stp_routes_max"]]

        # preparing MTP Data
        mtp_data = self.read_csv_from_s3(self.s3_object_path_dict["mtp_path"])
        mtp_data["stage"] = 999
        mtp_data.loc[
            mtp_data["execution_year_week"].isin(
                performance_stage_config_dict["stage2_config"]
            ),
            "stage",
        ] = 2
        mtp_data = mtp_data[mtp_data.stage != 999].reset_index(drop=True)
        mtp_data.loc[:, "station_pair"] = mtp_data["station"] + "-" + mtp_data["dsp"]

        # case where historical publish doesn't exists, the dcpm data would take its values from mtp data
        if dcpm_data.shape[0] > 0:
            # Merge all stage 2 configuration inputs
            dcpm_data = dcpm_data.merge(
                mtp_data,
                left_on=["station_pair", "year_week"],
                right_on=["station_pair", "execution_year_week"],
                how="left",
            )
        else:
            dcpm_data = mtp_data.copy()
            dcpm_data.rename(columns={"execution_year_week": "year_week"}, inplace=True)
            dcpm_data["dcpm_routes_max"] = dcpm_data["mtp_daily_routes"]

        dcpm_data = dcpm_data.merge(
            stp_data,
            left_on=["station_pair", "year_week"],
            right_on=["station_pair", "year_week"],
            how="left",
        )
        dcpm_data.loc[:, "mtp_daily_routes"].fillna(
            dcpm_data["dcpm_routes_max"]
        ).fillna(dcpm_data["lpt_daily_routes"]).fillna(
            dcpm_data["stp_routes_max"], inplace=True
        )
        dcpm_data.loc[:, "lpt_daily_routes"].fillna(
            dcpm_data["dcpm_routes_max"]
        ).fillna(dcpm_data["mtp_daily_routes"]).fillna(
            dcpm_data["stp_routes_max"], inplace=True
        )
        dcpm_data.loc[:, "stp_routes_max"].fillna(dcpm_data["dcpm_routes_max"]).fillna(
            dcpm_data["mtp_daily_routes"]
        ).fillna(dcpm_data["lpt_daily_routes"], inplace=True)

        if model_version == "Version 3.5 with DCPM Input":
            print("Running Model with DCPM Input")
            dcpm_data["daily_routes"] = dcpm_data["dcpm_routes_max"]
        elif model_version == "Version 3.5 with MTP Input":
            print("Running Model with MTP Input")
            dcpm_data["daily_routes"] = dcpm_data["mtp_daily_routes"]
        elif model_version == "Version 3.5 with STP Input":
            print("Running Model with STP Input")
            dcpm_data["daily_routes"] = dcpm_data["stp_routes_max"]
        else:
            print("Running with LPT Input")
            dcpm_data["daily_routes"] = dcpm_data["lpt_daily_routes"]

        dcpm_data = dcpm_data.groupby("station_pair").daily_routes.max().reset_index()
        return dcpm_data

    def merge_stage2_data(
        self, df_performance, performance_stage_config_dict, model_version, run_date
    ):
        """
        This function is used to inpute mtp forecast data to missing values for actuals with mtp forecasts in inference data

        Params:
        -----------------------
        valid_input: Dataframe
                Input data for inference
        mtp_data: Dataframe
                mtp forecast data
        mtp_execution_start_date: str
                mtp forecast start date
        mtp_execution_end_date: str
                mtp forecast end date

        Returns:
        -----------------------
        valid_input: Dataframe
                Input data with mtp forecasts substituted for actuals for stage 2 missing values

        Example:
        -----------------------
        valid_input = merge_stage2_data(valid_input, mtp_data)

        """
        mtp_data = self.create_stage2_data(
            performance_stage_config_dict, model_version, run_date
        )

        df_performance = df_performance.merge(
            mtp_data[["station_pair", "daily_routes"]], on="station_pair", how="left"
        )
        df_performance.rename(
            columns={"daily_routes": "completed_routes_2"}, inplace=True
        )

        df_performance["completed_routes_2"] = df_performance[
            "completed_routes_2"
        ].fillna(df_performance["completed_routes_1"])
        df_performance.loc[:, "final_route_target_2"] = np.nan
        df_performance.loc[:, "final_route_target_2"] = df_performance[
            "final_route_target_2"
        ].fillna(df_performance["final_route_target_1"])

        # df_performance = df_performance.fillna(df_performance.mean()).iloc[:, :-1]
        return df_performance

    def do_fallback_logic_for_transfers(self, targets):
        """
        This function is used to impute missing values with fallback logic
        """

        targets["completed_routes_2"].fillna(0, inplace=True)

        # Turning off features for targets prior to stage 2 config
        targets["vol_share_cycle1_1"] = 0
        targets["requested_routes_max_y_1"] = 0
        targets["completed_routes_max_y_3"] = 0
        targets["vin_active_rate_1"] = 0
        targets["vin_active_rate_3"] = 0

        targets.loc[
            targets.dsp_type != "Internal-Transfer", "completed_routes_max_1"
        ] = 10

        targets.loc[
            targets.dsp_type != "Internal-Transfer", "requested_routes_max_1"
        ] = 0

        targets.loc[
            targets.dsp_type != "Internal-Transfer", "completed_routes_max_3"
        ] = 0
        return targets

    def do_outlier_treatment_features(self, performance_input):
        """
        This function is used to treat outlier values of some important features of the model that drives capacity projections

        """
        #         # Outlier treatment for targets
        #         performance_input["label"] = np.where(
        #             (performance_input.station_pair_tenure_0 >= 7)
        #             & (performance_input.label < 15)
        #             & (performance_input.dsp_type != "DSP"),
        #             15,
        #             performance_input["label"],
        #         )
        # All Non-tenured DSPs would do minimum 15 routes post ramp-up periods
        performance_input["completed_routes_2"] = np.where(
            (performance_input.station_pair_tenure_0 >= 7)
            & (performance_input.completed_routes_2 < 15)
            & (performance_input.dsp_type != "DSP"),
            15,
            performance_input["completed_routes_2"],
        )

        # Outlier treatment for active da count: New-DSPs, Pop-ups ramping period
        performance_input["active_da_count_0"] = np.where(
            (performance_input.station_pair_tenure_0 <= 7)
            & (performance_input.completed_routes_2 <= 30)
            & (performance_input.dsp_type.isin(["Pop-Up", "New-DSP"])),
            30,
            performance_input["active_da_count_0"],
        )
        return performance_input

    def merge_transfers_data(
        self,
        performance_input,
        run_date,
        prediction_length,
        performance_stage_config_dict,
        model_version,
    ):
        """
        This function is used to merge DSP transfers, new launches, pop-ups, pinnacle to the performance input

        Params:
        -----------------------
        performance_input: Dataframe
                Input data for inference
        run_date: pd.to_datetime
                model run date
        mtp_execution_start_date: str
                mtp forecast start date
        mtp_execution_end_date: str
                mtp forecast end date

        Returns:
        -----------------------
        performance_input: Dataframe
                inference data with targets appended to the existing dsps

        Example:
        -----------------------
        performance_input = merge_transfers_data(performance_input,
                                                run_date,
                                                mtp_execution_start_date,
                                                mtp_execution_end_date)
        """

        mtp_data = self.create_stage2_data(
            performance_stage_config_dict, model_version, run_date
        )

        dsp_type_dict = {
            "Recruit to Expand": "Pop-Up",
            "Recruit to Expand - DSPx": "Pinnacle",
            "Recruit to Transfer": "Internal-Transfer",
            "Recruit to Offer": "New-DSP",
        }

        targets = self.read_csv_from_s3(self.s3_object_path_dict["transfers_path"])

        targets["target_launch_date"] = targets["target_launch_date"].astype(
            "datetime64[ns]"
        )
        targets = targets[~targets.station.isna()]
        targets = targets[targets.recruitment_type != "Transfer Outs"].reset_index(
            drop=True
        )

        targets = targets[
            (targets.target_launch_date >= pd.to_datetime(run_date - timedelta(49)))
            & (
                targets.target_launch_date
                <= pd.to_datetime(run_date + timedelta(7 * self.prediction_length))
            )
        ]  # We consider targets as targets under dsp_type features if they have launched later than 7 weeks prior to model run week

#         targets = targets[targets.cycle != "CYCLE_2"].reset_index(drop=True)

        targets.loc[:, "year"] = targets.target_launch_date.dt.year
        targets.loc[:, "week"] = targets.target_launch_date.dt.isocalendar().week
        targets.loc[:, "week"] = ["%02d" % x for x in targets.week]
        targets.loc[:, "year_week"] = (
            targets["year"].astype(str) + "-" + targets["week"].astype(str)
        )
        targets.loc[:, "dsp_type"] = targets["recruitment_type"].map(dsp_type_dict)
        targets.loc[:, "station_pair"] = targets["station"] + "-" + targets["dsp"]

        if (
            performance_stage_config_dict["prediction_year_week"][0].strip()[:4]
            == run_date.year + 1
        ):
            targets.loc[:, "prediction_week"] = (
                int(
                    performance_stage_config_dict["prediction_year_week"][0].strip()[
                        -2:
                    ]
                )
                + 52
            )
        else:
            targets.loc[:, "prediction_week"] = int(
                performance_stage_config_dict["prediction_year_week"][0].strip()[-2:]
            )

        prediction_date = run_date + timedelta(7 * prediction_length)
        targets["days_diff"] = (
            np.abs(
                (
                    pd.to_datetime(np.repeat(prediction_date, targets.shape[0]))
                    - targets.target_launch_date
                ).dt.days
            )
            // 7
        )

        targets.loc[:, "station_pair_tenure_0"] = np.where(
            ~(targets.dsp_type.isna()),
            targets["days_diff"],
            np.nan,
        )
        targets.loc[:, "station_pair_tenure_0"] = pd.to_numeric(
            targets["station_pair_tenure_0"]
        )
        targets.loc[:, "label"] = 1
        targets = targets.merge(
            mtp_data[["station_pair", "daily_routes"]], on="station_pair", how="left"
        )
        targets.rename(
            columns={
                "station": "station_code",
                "dsp": "dsp_code",
                "daily_routes": "completed_routes_2",
            },
            inplace=True,
        )

        # targets fallback logic
        targets = self.do_fallback_logic_for_transfers(targets)

        targets = targets[
            [
                "station_pair",
                "station_code",
                "dsp_code",
                "station_pair_tenure_0",
                "completed_routes_2",
                "vol_share_cycle1_1",
                "requested_routes_max_1",
                "vin_active_rate_1",
                "completed_routes_max_3",
                "completed_routes_max_1",
                "vin_active_rate_3",
                "dsp_type",
                "distance",
                "label",
            ]
        ].reset_index(drop=True)
        # targets = targets.drop_duplicates(keep="first")

        # Divye: We consider targets launch 7 week prior to run date as targets and hence update their dsp_type and station_pair_tenure

        # Modify existing DSP as targets based on above definition
        existing_targets = targets[
            targets.station_pair.isin(performance_input.station_pair)
        ].reset_index(drop=True)

        existing_dsp_type_dict = dict(
            zip(existing_targets.station_pair, existing_targets.dsp_type)
        )
        performance_input.loc[
            performance_input.station_pair.isin(existing_targets.station_pair),
            "dsp_type",
        ] = performance_input["station_pair"].map(existing_dsp_type_dict)

        existing_tenure_dict = dict(
            zip(existing_targets.station_pair, existing_targets.station_pair_tenure_0)
        )
        performance_input["station_pair_tenure_0_temp"] = performance_input[
            "station_pair"
        ].map(existing_tenure_dict)

        performance_input.loc[:, "station_pair_tenure_0"] = (
            performance_input["station_pair_tenure_0"]
            .fillna(performance_input["station_pair_tenure_0_temp"])
            .fillna(self.prediction_length)
        )

        performance_input.drop(columns=["station_pair_tenure_0_temp"], inplace=True)

        # Append new targets launching post run date
        new_targets = targets[
            ~targets.station_pair.isin(performance_input.station_pair)
        ].reset_index(drop=True)

        performance_input = pd.concat(
            [performance_input, new_targets], axis=0
        ).reset_index(drop=True)

        numeric_columns = performance_input.select_dtypes(
            include="number"
        ).columns.tolist()
        dsp_columns = numeric_columns + ["dsp_code"]
        dsp_metrics = (
            performance_input.reindex(columns=dsp_columns)
            .groupby(["dsp_code"])
            .median()
            .reset_index()
        )

        dsp_metrics = performance_input[["dsp_code"]].merge(
            dsp_metrics, on="dsp_code", how="left"
        )

        # First we replace the missing values using DSP code
        common_cols = list(
            set(performance_input.columns).intersection(set(dsp_metrics.columns))
        )
        for cols in common_cols:
            performance_input[cols].fillna(dsp_metrics[cols], inplace=True)

        numeric_columns = performance_input.select_dtypes(
            include="number"
        ).columns.tolist()
        station_columns = numeric_columns + ["station_code"]

        station_metrics = (
            performance_input.reindex(columns=station_columns)
            .groupby(["station_code"])
            .median()
            .reset_index()
        )
        station_metrics = performance_input[["station_code"]].merge(
            station_metrics, on="station_code", how="left"
        )

        # If DSP code is not available, then use Station code to replace remaining missing values
        common_cols = list(
            set(performance_input.columns).intersection(set(station_metrics.columns))
        )
        for cols in common_cols:
            performance_input[cols].fillna(station_metrics[cols], inplace=True)

        performance_input = self.do_outlier_treatment_features(performance_input)

        return performance_input

    def merge_exits_data(self, performance_input):
        """
        This function is used to replace the DSP type of existing DSP to exit status which are planned to exit prior to prediction week

        Params:
        -----------------------
        performance_input: Dataframe
                Input data for inference

        Returns:
        -----------------------
        performance_input: Dataframe
                Input data with dsp_type substituted for exit status

        Example:
        -----------------------
        performance_input = merge_exits_data(performance_input)

        """
        exits = self.read_csv_from_s3(self.s3_object_path_dict["exits_path"])
        exits.loc[:, "exits_status"] = "Exit" + "-" + exits.exit_status
        exits.rename(
            columns={"station": "station_code"},
            inplace=True,
        )
        exits.loc[:, "exit_date"] = pd.to_datetime(exits["exit_date"])

        performance_input["dsp_type"] = np.where(
            performance_input.station_pair.isin(exits.station_pair.unique()),
            "Exiting",
            performance_input.dsp_type,
        )
        # performance_input = performance_input[
        #     performance_input.dsp_type != "Exiting"
        # ].reset_index(drop=True)

        return performance_input

    def merge_transfer_out_data(
        self, performance_stage_config_dict, performance_input, run_date
    ):
        """
        This function is used to replace the DSP type of existing DSP to transfer out status which are planned to internal transfer prior to prediction week

        Params:
        -----------------------
        run_date: pd.to_datetime
                model run date
        performance_input: Dataframe
                Input data for inference

        Returns:
        -----------------------
        performance_input: Dataframe
                inference data with dsp_type Transfer Out for internal transfer DSPs

        Example:
        -----------------------
        performance_input = self.merge_transfers_out_data(run_date, performance_input)
        """

        transfers = self.read_csv_from_s3(self.s3_object_path_dict["transfers_path"])

        transfers["target_launch_date"] = pd.to_datetime(
            transfers["target_launch_date"]
        )
        transfers = transfers[transfers["recruitment_type"] == "Transfer Outs"]

        #         transfers = transfers[
        #             (transfers.target_launch_date + timedelta(1)).dt.isocalendar().week
        #             <= int(
        #                 performance_stage_config_dict["prediction_year_week"][0].strip()[-2:]
        #             )
        #         ]  # If a DSP is transferring out post prediction week, then it is not considered as a transfer out

        transfers = transfers[
            transfers.target_launch_date >= pd.to_datetime(run_date - timedelta(49))
        ].reset_index(drop=True)

        transfers = transfers[transfers.cycle != "CYCLE_2"].reset_index(drop=True)
        transfers.loc[:, "station_pair"] = transfers["station"] + "-" + transfers["dsp"]

        performance_input["dsp_type"] = np.where(
            performance_input.station_pair.isin(transfers.station_pair.unique()),
            "Transfer Outs",
            performance_input.dsp_type,
        )
        # Remove station pairs that are transfer outs
        # performance_input = performance_input[
        #     performance_input.dsp_type != "Transfer Outs"
        # ].reset_index(drop=True)

        return performance_input

    def merge_volume_forecast_data(
        self, performance_stage_config_dict, performance_input
    ):
        """
        This function is used to merge the station level lrp volume forecast data into performance data

        Params:
        ----------------------------------------------------------------
        None

        Returns:
        -------------------------
        performance_input: pd.DataFrame()
                preprocessed volume_forecast data called in create_input function
        """
        df = self.read_csv_from_s3(
            self.s3_object_path_dict["station_volume_forecast_path"]
        )
        df.loc[:, "lrp_station_vf_0"] = np.round(
            df["station_volume_forecast"] / df["work_days"]
        )
        df = df[
            df.year_week.isin(performance_stage_config_dict["prediction_year_week"])
        ].reset_index(drop=True)
        df = df.reindex(columns=["station_code", "lrp_station_vf_0"])

        performance_input = performance_input.merge(df, on=["station_code"], how="left")
        performance_input.loc[:, "lrp_station_vf_0"] = performance_input[
            "lrp_station_vf_0"
        ].fillna(performance_input["lrp_station_vf_0"].mean())

        return performance_input

    def add_da_hiring_signal(self, performance_input, performance_stage_config_dict, s3_dsp_data_upload_path):
        """
        This function is used to add da hiring signal from LPT model into the model pipeline

        """
        df_hiring = self.read_csv_from_s3(self.s3_object_path_dict["da_hiring_signal_path"])
        df_hiring = df_hiring.sort_values(by=["station_pair", "execution_year_week"]).reset_index(drop=True)

        prediction_week = performance_stage_config_dict["prediction_year_week"][0]

        df_hiring = df_hiring[df_hiring.execution_year_week == prediction_week].reset_index(drop=True)
        df_hiring = df_hiring[["station_pair", "da_hiring_signal"]]

        lrp_hiring = self.read_csv_from_s3(self.s3_object_path_dict["lrp_hiring_signal_path"])
        lrp_hiring["week"] = ["%02d" % x for x in lrp_hiring.week]
        lrp_hiring["year_week"] = lrp_hiring.year.astype(str) + '-' + lrp_hiring.week.astype(str)
        lrp_hiring = lrp_hiring[lrp_hiring.year_week == prediction_week].reset_index(drop=True)
        lrp_hiring = lrp_hiring[["station", "station_dsp_da_pool"]]

        volshare = self.read_parquet_from_s3(s3_dsp_data_upload_path)
        volshare["week"] = ["%02d" % x for x in volshare.week]
        volshare["year_week"] = volshare["year"].astype(str) + '-' + volshare["week"].astype(str)
        latest_volshare = volshare.groupby("station_pair")["year_week"].max().reset_index()
        volshare = volshare.merge(latest_volshare, on=["station_pair", "year_week"], how="inner")
        volshare = volshare[["station_pair", "station_code", "year_week", "vol_share_cycle1"]]
        volshare = volshare.merge(lrp_hiring, left_on="station_code", right_on="station", how="inner")
        volshare["lrp_hiring_signal"] = volshare["vol_share_cycle1"] * volshare["station_dsp_da_pool"]
        volshare = volshare[["station_pair", "lrp_hiring_signal"]]

        df_hiring = df_hiring.merge(volshare, on=["station_pair"], how="outer")
#         df_hiring["active_da_count_0"] = df_hiring[["lrp_hiring_signal", "da_hiring_signal"]].mean(axis=1) #testing
        df_hiring["active_da_count_0"] = df_hiring["da_hiring_signal"]
        df_hiring = df_hiring.drop(columns=["da_hiring_signal", "lrp_hiring_signal"])

        performance_input = performance_input.merge(
            df_hiring, on="station_pair", how="left"
        )
        performance_input["active_da_count_0"] = np.where(
            performance_input["active_da_count_0"].isna(),
            performance_input["active_da_count_3"],
            performance_input["active_da_count_0"],
        )

        return performance_input
    
    def create_station_pair_universe(self, run_date, performance_stage_config_dict):
        """
        This function is used to create Station-DSP Pair Universe working around edge cases to automate the issue for missing station pairs
        
        """
        station_univer_s3_path = f"etl_files/{run_date-timedelta(2)}/dsp_peak_scaling_station_universe.pqt"
        
        ramping_date = run_date - timedelta(7*7)
        run_year_week = str(run_date.year) +'-'+str(["%02d" % x for x in [run_date.isocalendar()[1]]][0])
        ramping_year_week = str(ramping_date.year) +'-'+str(["%02d" % x for x in [ramping_date.isocalendar()[1]]][0])

        input_data = self.read_parquet_from_s3(f"input_files/publish_date={run_date}/input_data.pqt")
        active_dsps = input_data[input_data.year_week == input_data.year_week.max()].station_pair.unique()
        all_dsps = input_data[input_data.year_week >= performance_stage_config_dict["stage5_config"][0]].station_pair.unique()
        print(f"Active_DSPs", len(active_dsps))
        print(f"All DSPs", len(all_dsps))

        transfers = self.read_csv_from_s3(f"etl_files/{run_date-timedelta(2)}/dsp_peak_scaling_transfers.csv000")
        transfers["target_launch_date"] = transfers["target_launch_date"].astype('datetime64[ns]')
        transfers["source_ds_last_route_date"] = transfers["source_ds_last_route_date"].astype('datetime64[ns]')
        transfers.loc[(transfers.recruitment_type == "Recruit to Expand") & (transfers.source_ds_last_route_date.isna()),
                      "source_ds_last_route_date"] = transfers.target_launch_date + pd.offsets.YearEnd()
        transfers["source_ds_last_route_date"] = transfers["source_ds_last_route_date"].astype('datetime64[ns]')
        transfers = transfers[transfers.target_launch_date.notna()]
        transfers["target_launch_year"] = transfers["target_launch_date"].dt.year
        transfers["target_launch_week"] = transfers["target_launch_date"].dt.isocalendar().week
        transfers["target_launch_week"] = ["%02d" % x for x in transfers["target_launch_week"]]
        transfers["year_week"] = transfers.target_launch_year.astype(str) + '-' + transfers.target_launch_week.astype(str)
        transfers = transfers[transfers.station_pair.notna()]
        transfers = transfers[transfers.year_week >= performance_stage_config_dict["stage5_config"][0]]
        print("Total targets in Output", transfers.station_pair.nunique())

        exits = self.read_csv_from_s3(f"etl_files/{run_date-timedelta(2)}/dsp_peak_scaling_exits.csv000")
        exits["exit_date"] = exits["exit_date"].astype('datetime64[ns]')
        exits["exit_year"] = exits["exit_date"].dt.year
        exits["exit_week"] = exits["exit_date"].dt.isocalendar().week
        exits["exit_week"] = ["%02d" % x for x in exits["exit_week"]]
        exits["year_week"] = exits.exit_year.astype(str) + '-' + exits.exit_week.astype(str)
        exits = exits[exits.exit_date.notna()]
        print("Total Exits in Output", exits.station_pair.nunique())

        forward_launches = transfers[transfers.year_week >= performance_stage_config_dict["run_year_week"]]
        transfer_outs = forward_launches[forward_launches.station_pair.isin(all_dsps)] #these are all transfer out DSPs
        forward_launches = forward_launches[~forward_launches.station_pair.isin(transfer_outs.station_pair.unique())]

        # there are some duplicates present since the station pairs can have multiple launches prior to model run date
        historical_launches = transfers[transfers.year_week < performance_stage_config_dict["run_year_week"]]
        df_temp = historical_launches.groupby(["station_pair"]).target_launch_date.max().reset_index()
        historical_launches = historical_launches.merge(df_temp, on=["station_pair", "target_launch_date"], how="inner")
        historial_launches = historical_launches[historical_launches.station_pair.isin(all_dsps)]

        last_work_week = input_data[input_data.year_week >= performance_stage_config_dict["stage5_config"][0]].groupby(["station_pair"]).agg(
            {"year_week": "max"}
        ).reset_index()
        latest_dsp_type = input_data[input_data.year_week >= performance_stage_config_dict["stage5_config"][0]].merge(last_work_week, 
                                                                                                                     on=["station_pair",
                                                                                                                         "year_week"], 
                                                                                                                      how="inner"
                                                                                                                     )[["station_pair", "dsp_type"]]

        sp_universe = input_data[input_data.year_week >= performance_stage_config_dict["stage5_config"][0]][["station_pair", 
                                                                                                             "station_code", 
                                                                                                             "dsp_code", 
                                                                                                             "dsp_type"]].drop_duplicates(keep='first').reset_index(drop=True)
        sp_universe = sp_universe.merge(latest_dsp_type, on=["station_pair", "dsp_type"], how="inner")
        df_temp = forward_launches.rename(columns={"station":"station_code", 
                                               "dsp":"dsp_code", 
                                               "recruitment_type":"dsp_type"})
        df_sp = pd.concat([sp_universe, df_temp[["station_pair", "station_code", "dsp_code", "dsp_type"]]]).reset_index(drop=True)
        df_sp["recruitment_type"] = np.where(df_sp.dsp_type.isin(['Recruit to Transfer', 
                                                                  'Recruit to Offer', 
                                                                  'Recruit to Expand - DSPx', 
                                                                  'Recruit to Expand', 
                                                                  'Transfer Outs']),
                                            df_sp.dsp_type,
                                            "DSP")
        

        ramping_df = historical_launches[historical_launches.year_week >= ramping_year_week].reset_index(drop=True)
        exits = exits[exits.station_pair.isin(df_sp.station_pair.unique())]
        exited_df = exits[exits.year_week <= run_year_week]

        ramping_map = dict(zip(ramping_df.station_pair, ramping_df.recruitment_type))
        dsp_map = dict(zip(historical_launches.station_pair, historical_launches.recruitment_type))
        transfer_out_map = dict(zip(transfer_outs.station_pair, transfer_outs.recruitment_type))
        exited_map = dict(zip(exited_df.station_pair, np.repeat("Exited", exited_df.station_pair.nunique())))
        exit_map = dict(zip(exits.station_pair, np.repeat("Exits", exits.station_pair.nunique())))

        #dsp_type
        df_sp.loc[df_sp.station_pair.isin(transfer_out_map.keys()), "dsp_type"] = df_sp["station_pair"].map(transfer_out_map)
        df_sp.loc[df_sp.station_pair.isin(ramping_map.keys()), "dsp_type"] = df_sp["station_pair"].map(ramping_map)
        df_sp.loc[df_sp.station_pair.isin(exited_map.keys()), "dsp_type"] = df_sp["station_pair"].map(exited_map)

        # recruitment_type
        df_sp.loc[df_sp.station_pair.isin(dsp_map.keys()), "recruitment_type"] = df_sp["station_pair"].map(dsp_map)
        df_sp.loc[df_sp.station_pair.isin(transfer_out_map.keys()), "recruitment_type"] = df_sp["station_pair"].map(transfer_out_map)
        df_sp.loc[df_sp.station_pair.isin(exits.station_pair.unique()), "recruitment_type"] = df_sp["station_pair"].map(exit_map)

        # Adding source station and destination station
        df_sp = df_sp.merge(historical_launches[["station_pair", "source_station"]], on=["station_pair"], how="left")
        ss_map = dict(zip(forward_launches.station_pair, forward_launches.source_station))
        df_sp.loc[df_sp.station_pair.isin(ss_map.keys()), "source_station"] = df_sp["station_pair"].map(ss_map)

        ss_map = dict(zip(transfer_outs.station_pair, transfer_outs.source_station))
        df_sp.loc[df_sp.station_pair.isin(ss_map.keys()), "source_station"] = df_sp["station_pair"].map(ss_map)

        df_sp["destination_station"] = df_sp.station_code
        df_sp.drop(columns=["station_code"], inplace=True)

        #defining the launch dates
        launch_map = dict(zip(historical_launches[historical_launches.recruitment_type.isin(['Recruit to Offer',
                                                                                             'Recruit to Expand - DSPx',
                                                                                             'Recruit to Expand'])].station_pair, 
                              historical_launches[historical_launches.recruitment_type.isin(['Recruit to Offer',
                                                                                             'Recruit to Expand - DSPx',
                                                                                             'Recruit to Expand'])].target_launch_date))
        df_sp.loc[df_sp.station_pair.isin(launch_map.keys()), "launch_date"] = df_sp["station_pair"].map(launch_map)

        launch_map = dict(zip(transfer_outs[transfer_outs.recruitment_type.isin(['Recruit to Offer',
                                                                                       'Recruit to Expand - DSPx',
                                                                                       'Recruit to Expand'])].station_pair,
                              transfer_outs[transfer_outs.recruitment_type.isin(['Recruit to Offer',
                                                                                       'Recruit to Expand - DSPx',
                                                                                       'Recruit to Expand'])].target_launch_date))
        df_sp.loc[df_sp.station_pair.isin(launch_map.keys()), "launch_date"] = df_sp["station_pair"].map(launch_map)

        launch_map = dict(zip(forward_launches[forward_launches.recruitment_type.isin(['Recruit to Offer',
                                                                                       'Recruit to Expand - DSPx',
                                                                                       'Recruit to Expand'])].station_pair,
                              forward_launches[forward_launches.recruitment_type.isin(['Recruit to Offer',
                                                                                       'Recruit to Expand - DSPx',
                                                                                       'Recruit to Expand'])].target_launch_date))
        df_sp.loc[df_sp.station_pair.isin(launch_map.keys()), "launch_date"] = df_sp["station_pair"].map(launch_map)

        # adding transfer in dates
        transfer_in_map = dict(zip(historical_launches[historical_launches.recruitment_type.isin(['Recruit to Transfer'])].station_pair,
                                  historical_launches[historical_launches.recruitment_type.isin(['Recruit to Transfer'])].target_launch_date))
        df_sp.loc[df_sp.station_pair.isin(transfer_in_map.keys()), "transfer_in_date"] = df_sp["station_pair"].map(transfer_in_map)

        transfer_in_map = dict(zip(transfer_outs[transfer_outs.recruitment_type.isin(['Recruit to Transfer'])].station_pair,
                                  transfer_outs[transfer_outs.recruitment_type.isin(['Recruit to Transfer'])].target_launch_date))
        df_sp.loc[df_sp.station_pair.isin(transfer_in_map.keys()), "transfer_in_date"] = df_sp["station_pair"].map(transfer_in_map)

        transfer_in_map = dict(zip(forward_launches[forward_launches.recruitment_type.isin(['Recruit to Transfer'])].station_pair,
                                  forward_launches[forward_launches.recruitment_type.isin(['Recruit to Transfer'])].target_launch_date))
        df_sp.loc[df_sp.station_pair.isin(transfer_in_map.keys()), "transfer_in_date"] = df_sp["station_pair"].map(transfer_in_map)

        # adding transfer out dates
        transfer_out_map = dict(zip(historical_launches[historical_launches.recruitment_type.isin(['Transfer Outs'])].station_pair,
                                  historical_launches[historical_launches.recruitment_type.isin(['Transfer Outs'])].target_launch_date))
        df_sp.loc[df_sp.station_pair.isin(transfer_out_map.keys()), "transfer_out_date"] = df_sp["station_pair"].map(transfer_out_map)

        transfer_out_map = dict(zip(transfer_outs[transfer_outs.recruitment_type.isin(['Transfer Outs'])].station_pair,
                                  transfer_outs[transfer_outs.recruitment_type.isin(['Transfer Outs'])].target_launch_date))
        df_sp.loc[df_sp.station_pair.isin(transfer_out_map.keys()), "transfer_out_date"] = df_sp["station_pair"].map(transfer_out_map)

        transfer_out_map = dict(zip(forward_launches[forward_launches.recruitment_type.isin(['Transfer Outs'])].station_pair,
                                  forward_launches[forward_launches.recruitment_type.isin(['Transfer Outs'])].target_launch_date))
        df_sp.loc[df_sp.station_pair.isin(transfer_out_map.keys()), "transfer_out_date"] = df_sp["station_pair"].map(transfer_out_map)

        # adding exit dates
        exit_date_map = dict(zip(exits.station_pair, exits.exit_date))
        df_sp.loc[df_sp.station_pair.isin(exit_date_map.keys()), "exit_date"] = df_sp["station_pair"].map(exit_date_map)

        exit_date_map = dict(zip(historical_launches[historical_launches.recruitment_type.isin(["Recruit to Expand"])].station_pair, 
                                 historical_launches[historical_launches.recruitment_type.isin(["Recruit to Expand"])].source_ds_last_route_date))
        df_sp.loc[df_sp.station_pair.isin(exit_date_map.keys()), "exit_date"] = df_sp["station_pair"].map(exit_date_map)

        exit_date_map = dict(zip(forward_launches[forward_launches.recruitment_type.isin(["Recruit to Expand"])].station_pair, 
                                 forward_launches[forward_launches.recruitment_type.isin(["Recruit to Expand"])].source_ds_last_route_date))
        df_sp.loc[df_sp.station_pair.isin(exit_date_map.keys()), "exit_date"] = df_sp["station_pair"].map(exit_date_map)
        
        dsp_type_dict = {
            "Recruit to Expand": "Pop-Up",
            "Recruit to Expand - DSPx": "Pinnacle",
            "Recruit to Transfer": "Internal-Transfer",
            "Recruit to Offer": "New-DSP",
        }
        df_sp.loc[df_sp.dsp_type.isin(['Recruit to Transfer', 
                                    'Recruit to Offer', 
                                    'Recruit to Expand - DSPx', 
                                    'Recruit to Expand']), "dsp_type"] = df_sp["dsp_type"].map(dsp_type_dict)
        
        self.write_parquet_to_s3(df_sp, station_univer_s3_path)

        logger.debug(f"Station DSP Pair Universe is created, total pairs = {df_sp.shape}")

        return df_sp

    def merge_lrp_station_universe_data(
        self, performance_input, performance_stage_config_dict, run_date
    ):
        """
        This function is used to add missing station pair data to DCM model output consider LRP station universe
        """
#         station_universe = self.read_csv_from_s3(
#             self.s3_object_path_dict["lrp_sp_universe_path"]
#         )
        
        station_universe = self.create_station_pair_universe(run_date, performance_stage_config_dict)
        
        station_universe["target_launch_date"] = (
            station_universe.launch_date.fillna(station_universe.transfer_in_date)
            .fillna(station_universe.transfer_out_date)
            .fillna(station_universe.exit_date)
        )
        station_universe["target_launch_date"] = pd.to_datetime(
            station_universe["target_launch_date"]
        )
        station_universe["station_code"] = station_universe.station_pair.str[:4]
        station_universe.loc[station_universe.dsp_type == "Exited", "dsp_type"] = (
            "Exiting"
        )

        if (
            performance_stage_config_dict["prediction_year_week"][0].strip()[:4]
            == run_date.year + 1
        ):
            station_universe.loc[:, "prediction_week"] = (
                int(
                    performance_stage_config_dict["prediction_year_week"][0].strip()[
                        -2:
                    ]
                )
                + 52
            )
        else:
            station_universe.loc[:, "prediction_week"] = int(
                performance_stage_config_dict["prediction_year_week"][0].strip()[-2:]
            )

        station_universe.loc[:, "station_pair_tenure_0"] = np.where(
            ~(station_universe.dsp_type.isna()),
            station_universe["prediction_week"]
            - (station_universe.target_launch_date + timedelta(1))
            .dt.isocalendar()
            .week,
            np.nan,
        )
        station_universe.loc[:, "station_pair_tenure_0"] = pd.to_numeric(
            station_universe["station_pair_tenure_0"]
        )

        #         station_universe.loc[:, "year"] = station_universe.target_launch_date.dt.year
        #         station_universe.loc[:, "week"] = station_universe.target_launch_date.dt.isocalendar().week
        #         station_universe.loc[:, "week"] = ["%02d" % x for x in station_universe.week]
        #         station_universe.loc[:, "year_week"] = (
        #             station_universe["year"].astype(str) + "-" + station_universe["week"].astype(str)
        #         )
        station_universe["label"] = 3
        station_universe = station_universe[
            ~station_universe.station_pair.isin(performance_input.station_pair)
        ]
        print(
            f"Shape of lrp station pair added to the output = {station_universe.shape}"
        )
        station_universe.drop(
            columns=[
                "target_launch_date",
                "prediction_week",
            ],
            inplace=True,
        )

        performance_input = pd.concat(
            [performance_input, station_universe], axis=0
        ).reset_index(drop=True)

        #         performance_input = performance_input[performance_input.station.isin(station_universe.station_pair.unique())].reset_index(drop=True) ## Rollback for next publish

        numeric_columns = performance_input.select_dtypes(
            include="number"
        ).columns.tolist()
        dsp_columns = numeric_columns + ["dsp_code"]
        dsp_metrics = (
            performance_input.reindex(columns=dsp_columns)
            .groupby(["dsp_code"])
            .median()
            .reset_index()
        )

        dsp_metrics = performance_input[["dsp_code"]].merge(
            dsp_metrics, on="dsp_code", how="left"
        )

        # First we replace the missing values using DSP code
        common_cols = list(
            set(performance_input.columns).intersection(set(dsp_metrics.columns))
        )
        for cols in common_cols:
            performance_input[cols].fillna(dsp_metrics[cols], inplace=True)

        numeric_columns = performance_input.select_dtypes(
            include="number"
        ).columns.tolist()
        station_columns = numeric_columns + ["station_code"]

        station_metrics = (
            performance_input.reindex(columns=station_columns)
            .groupby(["station_code"])
            .median()
            .reset_index()
        )
        station_metrics = performance_input[["station_code"]].merge(
            station_metrics, on="station_code", how="left"
        )

        # If DSP code is not available, then use Station code to replace remaining missing values
        common_cols = list(
            set(performance_input.columns).intersection(set(station_metrics.columns))
        )
        for cols in common_cols:
            performance_input[cols].fillna(station_metrics[cols], inplace=True)

        performance_input = self.do_outlier_treatment_features(performance_input)
        logger.debug(f"Merging LRP Station pair universe is completed")

        return performance_input

    def create_performance_input(
        self,
        valid_input,
        s3_dsp_data_upload_path,
        performance_stage_config_dict,
        local_download_path,
        run_date,
        prediction_length,
        s3_upload_path,
        s3_upload_flag,
        model_version,
    ):
        """
        This function is used to create performance input used by InferenceHandler to make final predictions

        Params:
        -----------------------
        valid_input: Dataframe
                staged performance data
        mtp_execution_start_date: str
                mtp forecast start date
        mtp_execution_end_date: str
                mtp forecast end date
        univariate_local_file_path: str
                file path to univariate forecasts for current year
        local_download_path: str
                file path to save predictions in local directory
        s3_upload_path: str
                file path to upload prediction in s3 location
        run_date: pd.to_datetime
                model run date

        Returns:
        -----------------------
        performance_input: Dataframe
                inference data used for inference

        Example:
        -----------------------
        performance_input = self.create_performance_input(params)
        """
        performance_input = self.merge_stage2_data(
            valid_input, performance_stage_config_dict, model_version, run_date
        )
        performance_input = self.impute_using_fallback_logic(performance_input)

        performance_input["dsp_tenure_week_0"] = (
            performance_input["dsp_tenure_week_1"]
            + int(performance_stage_config_dict["prediction_year_week"][0].strip()[-2:])
            - int(max(performance_stage_config_dict["stage1_config"]).strip()[-2:])
        )
        performance_input["station_pair_tenure_0"] = (
            performance_input["station_pair_tenure_1"]
            + int(performance_stage_config_dict["prediction_year_week"][0].strip()[-2:])
            - int(max(performance_stage_config_dict["stage1_config"]).strip()[-2:])
        )

        performance_input = self.add_da_hiring_signal(
            performance_input, performance_stage_config_dict, s3_dsp_data_upload_path
        )

        performance_input = self.merge_transfers_data(
            performance_input,
            run_date,
            prediction_length,
            performance_stage_config_dict,
            model_version,
        )
        performance_input = self.merge_exits_data(performance_input)
        performance_input = self.merge_transfer_out_data(
            performance_stage_config_dict, performance_input, run_date
        )
        performance_input = self.merge_volume_forecast_data(
            performance_stage_config_dict, performance_input
        )
        performance_input = self.merge_lrp_station_universe_data(
            performance_input, performance_stage_config_dict, run_date
        )

        dsp_cat = pd.get_dummies(performance_input["dsp_type"], dtype=float)
        dsp_type_list = [
            "DSP 2.0",
            "DSP 1.0",
            "Walker",
            "Internal-Transfer",
            "Pinnacle",
            "Pop-Up",
            "New-DSP",
            "Transfer Outs",
            "Exiting",
        ]
        if len(dsp_cat.columns) < len(dsp_type_list):
            missing_cols = set(dsp_type_list).difference(dsp_cat)
            for cols in missing_cols:
                dsp_cat[cols] = 0
                
        performance_input = pd.concat([performance_input, dsp_cat], axis=1)

        performance_input = performance_input.drop_duplicates(keep="first").reset_index(
            drop=True
        )

        df_temp = (
            performance_input.groupby("station_pair")
            .agg({"active_da_count_1": "count"})
            .reset_index()
            .sort_values(by=["active_da_count_1"], ascending=False)
        )
        df_temp = df_temp[df_temp.active_da_count_1 > 1]

        performance_input = performance_input[
            ~performance_input.station_pair.isin(df_temp.station_pair.unique())
        ]
        #         performance_input.loc[performance_input.label.isna(), "label"] = 2
        df_temp = performance_input.select_dtypes(include="number")
        df_temp = df_temp.fillna(df_temp.median())
        performance_input = performance_input.fillna(df_temp)

        #         if performance_input.shape[0] != performance_input.station_pair.nunique():
        #             raise Exception("Station pair duplicates exists")

        performance_input.to_parquet(local_download_path, index=False)
        if s3_upload_flag:
            self.upload_to_s3(
                local_download_path=local_download_path, s3_upload_path=s3_upload_path
            )

        logger.debug(f"Successfully created performance input data with shape {performance_input.shape}")


class ModelTrainer(DataHandler):
    def __init__(self, bucket=None):
        DataHandler.__init__(self, bucket)
        self.X_train = pd.DataFrame()
        self.y_train = pd.Series()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()

    def model_performance_report(self, y_true, y_pred):
        """
        This function calculates the performance of any regression model
        Params:
        -----------------------
        y_true: array
                true labels
        y_pred: array
                predicted labels

        Returns:
        -----------------------
        result: dict
                model performance metrics

        Example:
        -----------------------
        ModelTrainer.model_performance_report(params)
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = np.mean(np.abs(y_pred - y_true))
        mape = np.mean(np.abs(y_pred - y_true) / y_true)
        wape = np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)
        r2 = r2_score(y_true, y_pred)

        print(f"Mean Squared Error {np.round(mse, 6)}")
        print(f"Mean Absoluted Percentage Error {np.round(100*mape, 6)}")
        print(f"Weighted Absolute Percentage Error {np.round(100*wape, 6)}")
        print(f"Model R-Square {r2}")

        result = {"mse": mse, "mae": mae, "mape": mape, "wape": wape, "r2": r2}

        return result

    def do_train_test_transformation_split(
        self,
        train_data_path,
        model_features,
        local_power_transformer_path,
        local_scaler_path,
        s3_power_transformer_path,
        s3_scaler_upload_path,
        s3_upload_flag,
    ):
        """
        This function is used to prepare the training and testing datasets for trying out different machine learning models.

        Params:
        -----------------------
        train_data_path: str
                local file path to training_data
        model_features: list
                list of model features used for training

        Returns:
        -----------------------
        Run logs

        Example:
        -----------------------
        ModelTrainer.do_train_test_transformation_split(params)
        """
        data = pd.read_parquet(train_data_path)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.dropna().reset_index(drop=True)

        train_data = data.reindex(columns=model_features)
        train_data = (
            train_data[train_data["label"] >= 0]
            .dropna(axis=1, how="all")
            .reset_index(drop=True)
        )
        train_data = train_data.drop_duplicates(keep="first").reset_index(drop=True)
        train_data = train_data.dropna().reset_index(drop=True)
        df_train, df_test = train_test_split(train_data, test_size=0.3, random_state=6)
        self.df_train = df_train
        self.df_test = df_test

        y_train = df_train["label"].reset_index(drop=True)
        y_test = df_test["label"].reset_index(drop=True)

        X_manager_train = (
            df_train.drop(columns=["label"]).astype(float).reset_index(drop=True)
        )
        X_team_test = (
            df_test.drop(columns=["label"]).astype(float).reset_index(drop=True)
        )

        power_transformer = PowerTransformer()
        scaler = StandardScaler()

        X_train = power_transformer.fit_transform(X_manager_train)
        X_train = scaler.fit_transform(X_train)

        X_test = power_transformer.transform(X_team_test)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=X_manager_train.columns)
        X_test = pd.DataFrame(X_test, columns=X_team_test.columns)

        try:
            os.makedirs(
                local_scaler_path.split("/dsp_peak_scaling_linear_scaler.pkl")[0]
            )
        except:
            logger.debug("Directory already exists")

        with open(local_power_transformer_path, "wb") as f:
            joblib.dump(power_transformer, f)
        with open(local_scaler_path, "wb") as f:
            joblib.dump(scaler, f)

        if s3_upload_flag:
            self.upload_to_s3(
                local_download_path=local_power_transformer_path,
                s3_upload_path=s3_power_transformer_path,
            )
            self.upload_to_s3(
                local_download_path=local_scaler_path,
                s3_upload_path=s3_scaler_upload_path,
            )

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        logger.debug("Created train and test split along with transformations")

    def call_model_selection_helper(
        self,
        model,
        train_data_path,
        local_model_path,
        local_train_file_path,
        local_test_file_path,
        local_model_performance_path,
        s3_model_upload_path,
        s3_upload_flag,
        model_name="default",
        run="partial",
    ):
        data = pd.read_parquet(train_data_path)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.dropna().reset_index(drop=True)

        self.X_train = self.X_train.astype("float32")
        self.X_train.replace([np.inf, -np.inf], 0, inplace=True)

        self.X_test = self.X_test.astype("float32")
        self.X_test.replace([np.inf, -np.inf], 0, inplace=True)

        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        logger.debug(
            f"Model Evaluation Metrics for {model_name} on training dataset and validation dataset"
        )
        metric_train = self.model_performance_report(self.y_train, y_pred_train)
        print()

        metric_test = self.model_performance_report(self.y_test, y_pred_test)
        print()

        if run == "full":
            df_metric_train = pd.DataFrame.from_dict(
                metric_train, columns=["train_values"], orient="index"
            ).reset_index()
            df_metric_train.rename(columns={"index": "metric_name"}, inplace=True)

            df_metric_test = pd.DataFrame.from_dict(
                metric_test, columns=["test_values"], orient="index"
            ).reset_index()
            df_metric_test.rename(columns={"index": "metric_name"}, inplace=True)

            df_metric = df_metric_train.merge(
                df_metric_test, on=["metric_name"], how="inner"
            )
            df_metric["selected_model"] = model_name

            df_train = self.df_train.join(
                data[["dsp_type", "dsp_code", "station_code", "station_pair"]]
            )
            df_train["predicted_label"] = y_pred_train

            df_test = self.df_test.join(
                data[["dsp_type", "dsp_code", "station_code", "station_pair"]]
            )
            df_test["predicted_label"] = y_pred_test

            df_metric.to_csv(local_model_performance_path, index=False)
            try:
                os.makedirs(
                    local_model_path.split("/dsp_peak_scaling_linear_model.pkl")[0]
                )
            except:
                logger.debug("Directory already exists")

            with open(local_model_path, "wb") as f:
                joblib.dump(model, f)

            df_train.to_csv(local_train_file_path, index=False)
            df_test.to_csv(local_test_file_path, index=False)

            if s3_upload_flag:
                self.upload_to_s3(
                    local_download_path=local_model_path,
                    s3_upload_path=s3_model_upload_path,
                )

            logger.debug(
                f"Dumped trained model pickle file in folder: {local_model_path}"
            )
            logger.debug(f"Training {model_name} is completed")
            logger.debug(
                f"Shape of df_train = {df_train.shape}, Shape of df_test = {df_test.shape}, Shape of data = {data.shape}"
            )
        else:
            return metric_test

    def do_model_training(
        self,
        train_data_path,
        local_model_path,
        local_train_file_path,
        local_test_file_path,
        local_model_performance_path,
        s3_model_upload_path,
        s3_upload_flag,
    ):

        # Linear Model
        linear_grid = {}
        linear_regressor = LinearRegression(n_jobs=-1)
        linear_model_cv = GridSearchCV(
            linear_regressor,
            linear_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        linear_model = linear_model_cv.fit(self.X_train, self.y_train).best_estimator_
        linear_metric = self.call_model_selection_helper(
            linear_model,
            train_data_path,
            local_model_path,
            local_train_file_path,
            local_test_file_path,
            local_model_performance_path,
            s3_model_upload_path,
            s3_upload_flag,
            model_name="linear_model",
        )

        # XGB Model
        xgb_grid = {
            "booster": ["gblinear"],
            "objective": ["reg:squarederror"],
            "n_estimators": [150, 300],
            "learning_rate": [0.01, 0.03],
            "reg_lambda": np.arange(0.0, 1.0, 0.1),
            "reg_alpha": [0.001, 0.1, 1, 10, 100],
            "eval_metric": ["rmse"],
        }
        xgb_regressor = XGBRegressor(
            random_state=6,
            n_jobs=-1,
        )
        xgb_regressor_cv = GridSearchCV(
            xgb_regressor,
            xgb_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        gblinear_model = xgb_regressor_cv.fit(
            self.X_train, self.y_train
        ).best_estimator_
        gblinear_metric = self.call_model_selection_helper(
            gblinear_model,
            train_data_path,
            local_model_path,
            local_train_file_path,
            local_test_file_path,
            local_model_performance_path,
            s3_model_upload_path,
            s3_upload_flag,
            model_name="gblinear_model",
        )

        # ElasticNet Model
        elastic_grid = {
            "max_iter": [1000, 5000],
            "alpha": [0.01, 0.1, 1, 10, 100],
            "l1_ratio": np.arange(0.0, 1.0, 0.1),
        }
        elasticnet_regressor = ElasticNet(random_state=6)
        elasticnet_regressor_cv = GridSearchCV(
            elasticnet_regressor,
            elastic_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        elastic_model = elasticnet_regressor_cv.fit(
            self.X_train, self.y_train
        ).best_estimator_
        elastic_metric = self.call_model_selection_helper(
            elastic_model,
            train_data_path,
            local_model_path,
            local_train_file_path,
            local_test_file_path,
            local_model_performance_path,
            s3_model_upload_path,
            s3_upload_flag,
            model_name="elasticnet_model",
        )
        
        d = {
            linear_model: linear_metric["wape"],
            gblinear_model: gblinear_metric["wape"],
            elastic_model: elastic_metric["wape"]
        }
        self.call_model_selection_helper(
            min(d, key=d.get),
            train_data_path,
            local_model_path,
            local_train_file_path,
            local_test_file_path,
            local_model_performance_path,
            s3_model_upload_path,
            s3_upload_flag,
            model_name=min(d, key=d.get),
            run="full",
        )


class InferenceHandler(DataHandler):
    def __init__(
        self, run_date, s3_path_dict, performance_stage_config_dict, bucket=None
    ):
        DataHandler.__init__(self, bucket)
        self.run_date = run_date
        self.performance_stage_config_dict = performance_stage_config_dict
        self.s3_path_dict = s3_path_dict

    def make_explanations(self, model, df):
        """
        This function is used to make shap explanations for model features

        """
        explainer = shap.explainers.Linear(model, df)
        shap_values = explainer(df)
        return shap_values

    def make_corrections(self, performance_data, prediction_length, run_date):
        """
        This function is used to make corrections to edge cases of the model output

        """
        # The following code prevents model to generate negative forecasts
        performance_data["predictions_with_incentives"] = np.where(
            performance_data["predictions_with_incentives"] < 0,
            5,
            performance_data["predictions_with_incentives"],
        )
        performance_data["predictions_no_incentives"] = np.where(
            performance_data["predictions_no_incentives"] < 0,
            5,
            performance_data["predictions_no_incentives"],
        )
        performance_data["predictions_with_overrides"] = np.where(
            performance_data["launch_date"].astype('datetime64[ns]') >= '2023-09-01', 
            np.maximum(22, performance_data["predictions_with_overrides"]), 
            performance_data["predictions_with_overrides"]
        )
        performance_data["predictions_with_overrides"] = np.where(
            performance_data["launch_date"].astype('datetime64[ns]') >= '2023-09-01', 
            np.minimum(35, performance_data["predictions_with_overrides"]), 
            performance_data["predictions_with_overrides"]
        )
        performance_data = performance_data[
            (performance_data.launch_date.isna()) | (performance_data.launch_date.astype('datetime64[ns]') <= str(run_date+timedelta(prediction_length*7)))
        ]
        performance_data["predictions_with_overrides"] = np.where(
            performance_data.station_pair_tenure_0.isin([0,1,2,3]), 5,
            np.where(
                performance_data.station_pair_tenure_0.isin([4,5]), 10,
                np.where(performance_data.station_pair_tenure_0.isin([6]), 15,
                    np.where(
                        performance_data.station_pair_tenure_0.isin([7]), 20, 
                        performance_data["predictions_with_overrides"]
                    ) 
                )
            )
        )
        return performance_data

    def calculate_steady_state_routes(self, input_data_path):
        """
        This function is used to calculate steady state routes based on this SIM: https://issues.amazon.com/issues/P121943089

        """
        input_data = self.read_parquet_from_s3(input_data_path)
        input_data = input_data[
            ["year_week", "year", "week", "station_pair", "completed_routes_max"]
        ]
        steady_state_weeks = [str(i) for i in range(30, 40)]
        if (
            int(
                self.performance_stage_config_dict["prediction_year_week"][0].split(
                    "-"
                )[1]
            )
            < 40
        ):
            steady_state_year = (
                int(
                    self.performance_stage_config_dict["prediction_year_week"][0].split(
                        "-"
                    )[0]
                )
                - 1
            )
            input_data = input_data[
                (input_data.year == steady_state_year)
                & (input_data.week.isin(steady_state_weeks))
            ]
            input_data = (
                input_data.groupby(["station_pair"])
                .agg({"completed_routes_max": "mean"})
                .reset_index()
            )

        return input_data

    def calculate_peak_routes(self, input_data_path):
        """
        This function is used to return peak routes of last year

        """
#         input_data = self.read_parquet_from_s3(input_data_path)
#         peak_predictions = self.read_parquet_from_s3("inferences_v4/publish_date=2023-09-20/region=NA/prediction_year_week=2023-50/dsp_peak_scaling_linear_predictions.pqt")
        tenet_inputs = self.read_csv_from_s3("etl_files/dsp_peak_scaling_tenets_inputs.csv")
        tenet_inputs["completed_routes_max"] = np.maximum(tenet_inputs.peak_prediction_75, tenet_inputs.peak_actual_75)
        
#         input_data = input_data[
#             ["year_week", "year", "week", "station_pair", "completed_routes_max"]
#         ]
#         peak_year = (
#             int(
#                 self.performance_stage_config_dict["prediction_year_week"][0].split(
#                     "-"
#                 )[0]
#             )
#             - 1
#         )
#         peak_weeks = [str(50)]
#         input_data = input_data[
#             (input_data.year == peak_year) & (input_data.week.isin(peak_weeks))
#         ]
#         input_data = (
#             input_data.groupby(["station_pair"])
#             .agg({"completed_routes_max": "mean"})
#             .reset_index()
#         )
#         input_data = input_data.merge(peak_predictions[["station_pair", "predictions_with_incentives"]], on="station_pair", how="left")
#         input_data["predictions_with_incentives"] = input_data["predictions_with_incentives"].fillna(input_data.completed_routes_max)
#         input_data["completed_routes_max"] = np.maximum(input_data.completed_routes_max, input_data.predictions_with_incentives)
#         input_data.drop(columns=["predictions_with_incentives"], inplace=True)
        return tenet_inputs[["station_pair", "completed_routes_max"]]

    def make_overrides(self, run_date, performance_data, input_data_path):
        """
        This function is used to calculate upper bound predictions on performance input predictions with incentives based on tenets
        """

        br_data = self.read_csv_from_s3(self.s3_path_dict["br_path"])
        high_risk_dsps = (
            br_data[br_data.business_review == "High Risk"].dsp.unique().tolist()
        )
        medium_risk_dsps = (
            br_data[br_data.business_review == "Intermediate Risk"]
            .dsp.unique()
            .tolist()
        )
        low_risk_dsps = (
            br_data[br_data.business_review == "Low Risk"].dsp.unique().tolist()
        )

        quality_score = self.read_csv_from_s3(self.s3_path_dict["dsp_quality_path"])
        quality_score.loc[:, "dsp_quality_score"] = (
            quality_score["quality_score_num"] / quality_score["quality_score_denom"]
        )
        quality_score = quality_score[["dsp_code", "year_week", "dsp_quality_score"]]
        latest_qs = (
            quality_score.groupby(["dsp_code"]).agg({"year_week": "max"}).reset_index()
        )
        quality_score = quality_score.merge(latest_qs, on=["dsp_code", "year_week"])

        low_quality_dsps = (
            quality_score[quality_score.dsp_quality_score == 0.0]
            .dsp_code.unique()
            .tolist()
        )

        performance_data["predictions_with_overrides"] = performance_data[
            "predictions_with_incentives"
        ]

        # Tenet Driven OverRides - year around
        # 0. We believe that all DSPs 7 weeks post their launch date would be running minimum 20 routes
        performance_data["predictions_with_overrides"] = np.where(
            performance_data["station_pair_tenure_0"] > 7,
            np.maximum(20, performance_data["predictions_with_overrides"]),
            performance_data["predictions_with_overrides"],
        )

        # 1. We believe that high reliability DSPs (above average) can scale minimum to what they did last year
        performance_data["predictions_with_overrides"] = np.where(
            performance_data["reliability_1"]
            > performance_data["reliability_1"].median(),
            np.maximum(
                performance_data["predictions_with_overrides"],
                performance_data["completed_routes_max_3"],
            ),
            performance_data["predictions_with_overrides"],
        )
        
#       Closed out the peak tenets based on 5/6 discussion
#         if int(
#             self.performance_stage_config_dict["prediction_year_week"][0].split("-")[1]
#         ) in [48, 49, 50, 51, 52]:
#             # 2. We believe Pop-Ups are able to do minimum 27 routes this year
#             performance_data["predictions_with_overrides"] = np.where(
#                 performance_data["dsp_type"] == "Pop-Up",
#                 performance_data["predictions_with_overrides"].clip(lower=27.0),
#                 performance_data["predictions_with_overrides"],
#             )

#             # 3. We believe that all DSPs will be flexing up by 5% to meet positive fluctuations in volume
#             performance_data["predictions_with_overrides"] = (
#                 performance_data["predictions_with_overrides"] * 1.05
#             )

        steady_state_tenet_period = [
            i for i in range(5, 40) if i not in [25, 26, 27, 28, 29, 30]
        ]
        if (
            int(
                self.performance_stage_config_dict["prediction_year_week"][0].split(
                    "-"
                )[1]
            )
            in steady_state_tenet_period
        ):
            # 4. We believe that high risk business review DSPs are capped at 15 routes
            performance_data["predictions_with_overrides"] = np.where(
                performance_data["dsp_code"].isin(high_risk_dsps),
                np.minimum(15, performance_data["predictions_with_overrides"]),
                performance_data["predictions_with_overrides"],
            )

            # 5. We believe that medium risk business review DSPs should not scale above Q3’23- steady state route avg in the quarter.
            # To calculate steady state route avg, we can use average of route actuals between Wk 30-39
            steady_state_routes = self.calculate_steady_state_routes(input_data_path)
            performance_data = performance_data.merge(
                steady_state_routes, on=["station_pair"], how="left"
            )
            performance_data.loc[:, "completed_routes_max"] = performance_data[
                "completed_routes_max"
            ].fillna(performance_data["predictions_with_overrides"])

            september_launch = (
                run_date - datetime(run_date.year - 1, 9, 1).date()
            ).days // 7

            performance_data["predictions_with_overrides"] = np.where(
                (~performance_data["dsp_code"].isin(high_risk_dsps))
                & (performance_data["dsp_code"].isin(low_quality_dsps)),
                np.maximum(22, performance_data["completed_routes_max"]),
                performance_data["predictions_with_overrides"],
            )

            performance_data.drop(columns=["completed_routes_max"], inplace=True)

            # 6. We believe that low risk dsps are doing minimum of ( 0.75*Peak (wk 50)completed routes max , override predictions)
            peak_routes = self.calculate_peak_routes(input_data_path)
            performance_data = performance_data.merge(
                peak_routes, on=["station_pair"], how="left"
            )
            performance_data.loc[:, "completed_routes_max"] = performance_data[
                "completed_routes_max"
            ].fillna(performance_data["predictions_with_overrides"])
            performance_data["predictions_with_overrides"] = np.where(
                (~performance_data["dsp_code"].isin(high_risk_dsps))
                & (~performance_data["dsp_code"].isin(low_quality_dsps)),
                np.minimum(
                    performance_data["predictions_with_overrides"],
                    0.75 * performance_data["completed_routes_max"],
                ),
                performance_data["predictions_with_overrides"],
            )
            performance_data.drop(columns=["completed_routes_max"], inplace=True)
            
            # 7. We believe that DSPs launched post september of last year and are medium and low risk DSPs would do maximum of 35
            performance_data["predictions_with_overrides"] = np.where(
                (~performance_data["dsp_code"].isin(high_risk_dsps))
#                 & (performance_data["dsp_code"].isin(low_quality_dsps))
                & (performance_data["station_pair_tenure_0"] <= september_launch),
                np.minimum(35, performance_data["predictions_with_overrides"]),
                performance_data["predictions_with_overrides"],
            )

        performance_data.loc[:, "predictions_with_overrides"] = np.where(
            performance_data["predictions_with_overrides"] < 0,
            0,
            performance_data["predictions_with_overrides"],
        )

        performance_data.loc[:, "predictions_with_overrides"] = np.round(
            performance_data["predictions_with_overrides"], 6
        )

        return performance_data

    def merge_auxillary_data(self, performance_data, s3_object_path_dict, run_date):
        """
        This function is used to merge target auxillary data with performance data
        """

        targets = self.read_csv_from_s3(s3_object_path_dict["transfers_path"])

        targets.loc[:, "target_launch_date"] = pd.to_datetime(
            targets["target_launch_date"]
        )
        targets.loc[:, "source_ds_last_route_date"] = pd.to_datetime(
            targets["source_ds_last_route_date"]
        )
        targets = targets[~targets.station.isna()]
        targets = targets[
            targets.target_launch_date >= pd.to_datetime(run_date - timedelta(365))
        ]
        targets.loc[:, "station_pair"] = targets["station"] + "-" + targets["dsp"]

        exits = self.read_csv_from_s3(s3_object_path_dict["exits_path"])
        exits.loc[:, "exit_date"] = pd.to_datetime(exits["exit_date"]).dt.date

        recruitment_type_dict = dict(
            zip(targets.station_pair, targets.recruitment_type)
        )
        source_station_dict = dict(zip(targets.station_pair, targets.source_station))
        destination_station_dict = dict(zip(targets.station_pair, targets.station))

        launch_date_dict = dict(
            zip(
                targets[
                    targets.recruitment_type.isin(
                        [
                            "Recruit to Offer",
                            "Recruit to Expand",
                            "Recruit to Expand - DSPx",
                        ]
                    )
                ].station_pair,
                targets[
                    targets.recruitment_type.isin(
                        [
                            "Recruit to Offer",
                            "Recruit to Expand",
                            "Recruit to Expand - DSPx",
                        ]
                    )
                ].target_launch_date,
            )
        )
        transfer_in_date_dict = dict(
            zip(
                targets[
                    targets.recruitment_type.isin(["Recruit to Transfer"])
                ].station_pair,
                targets[
                    targets.recruitment_type.isin(["Recruit to Transfer"])
                ].target_launch_date,
            )
        )
        transfer_out_date_dict = dict(
            zip(
                targets[targets.recruitment_type.isin(["Transfer Outs"])].station_pair,
                targets[
                    targets.recruitment_type.isin(["Transfer Outs"])
                ].target_launch_date,
            )
        )
        exit_date_dict = dict(zip(exits.station_pair, exits.exit_date))

        popup_exits = dict(
            zip(
                targets[
                    targets.recruitment_type.isin(["Recruit to Expand"])
                ].station_pair,
                targets[
                    targets.recruitment_type.isin(["Recruit to Expand"])
                ].source_ds_last_route_date,
            )
        )

        performance_data.loc[:, "recruitment_type"] = performance_data[
            "station_pair"
        ].map(recruitment_type_dict)
        performance_data.loc[:, "source_station"] = performance_data[
            "station_pair"
        ].map(source_station_dict)
        performance_data.loc[:, "destination_station"] = performance_data[
            "station_pair"
        ].map(destination_station_dict)
        performance_data.loc[:, "launch_date"] = performance_data["station_pair"].map(
            launch_date_dict
        )
        performance_data.loc[:, "transfer_in_date"] = performance_data[
            "station_pair"
        ].map(transfer_in_date_dict)
        performance_data.loc[:, "transfer_out_date"] = performance_data[
            "station_pair"
        ].map(transfer_out_date_dict)

        performance_data.loc[:, "exit_date"] = performance_data["station_pair"].map(
            popup_exits
        )

        performance_data.loc[:, "exit_date"] = performance_data["station_pair"].map(
            exit_date_dict
        )

        return performance_data

    def override_auxillary_data(self, performance_data, s3_object_path_dict, run_date):
        """
        This function is used to fill missing auxillary information using lrp station pair universe
        """
        station_univer_s3_path = f"etl_files/{run_date-timedelta(2)}/dsp_peak_scaling_station_universe.pqt"
        station_universe = self.read_parquet_from_s3(station_univer_s3_path)
        station_universe["target_launch_date"] = (
            station_universe.launch_date.fillna(station_universe.transfer_in_date)
            .fillna(station_universe.transfer_out_date)
            .fillna(station_universe.exit_date)
        )
        station_universe["target_launch_date"] = pd.to_datetime(
            station_universe["target_launch_date"]
        )
        station_universe["station_code"] = station_universe.station_pair.str[:4]
        station_universe.loc[station_universe.dsp_type == "Exited", "dsp_type"] = (
            "Exiting"
        )

        recruitment_type_dict = dict(
            zip(station_universe.station_pair, station_universe.recruitment_type)
        )
        source_station_dict = dict(
            zip(station_universe.station_pair, station_universe.source_station)
        )
        destination_station_dict = dict(
            zip(station_universe.station_pair, station_universe.destination_station)
        )

        launch_date_dict = dict(
            zip(
                station_universe[
                    station_universe.recruitment_type.isin(
                        [
                            "Recruit to Offer",
                            "Recruit to Expand",
                            "Recruit to Expand - DSPx",
                        ]
                    )
                ].station_pair,
                station_universe[
                    station_universe.recruitment_type.isin(
                        [
                            "Recruit to Offer",
                            "Recruit to Expand",
                            "Recruit to Expand - DSPx",
                        ]
                    )
                ].target_launch_date,
            )
        )
        transfer_in_date_dict = dict(
            zip(
                station_universe[
                    station_universe.recruitment_type.isin(["Recruit to Transfer"])
                ].station_pair,
                station_universe[
                    station_universe.recruitment_type.isin(["Recruit to Transfer"])
                ].target_launch_date,
            )
        )
        transfer_out_date_dict = dict(
            zip(
                station_universe[
                    station_universe.recruitment_type.isin(["Transfer Outs"])
                ].station_pair,
                station_universe[
                    station_universe.recruitment_type.isin(["Transfer Outs"])
                ].target_launch_date,
            )
        )
        exit_date_dict = dict(
            zip(
                station_universe[
                    station_universe.dsp_type.isin(["Exiting"])
                ].station_pair,
                station_universe[
                    station_universe.dsp_type.isin(["Exiting"])
                ].target_launch_date,
            )
        )

        performance_data.loc[:, "recruitment_type"] = performance_data[
            "station_pair"
        ].map(recruitment_type_dict)
        performance_data.loc[:, "source_station"] = performance_data[
            "station_pair"
        ].map(source_station_dict)
        performance_data.loc[:, "destination_station"] = performance_data[
            "station_pair"
        ].map(destination_station_dict)
        performance_data.loc[:, "launch_date"] = performance_data["station_pair"].map(
            launch_date_dict
        )
        performance_data.loc[:, "transfer_in_date"] = performance_data[
            "station_pair"
        ].map(transfer_in_date_dict)
        performance_data.loc[:, "transfer_out_date"] = performance_data[
            "station_pair"
        ].map(transfer_out_date_dict)

        performance_data.loc[:, "exit_date"] = performance_data["station_pair"].map(
            exit_date_dict
        )
        return performance_data

    def do_quality_checks(self, df):
        """
        This function is used to do quality checks on the output file

        """
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        df.loc[:, numeric_columns] = np.round(df[numeric_columns], 6).astype("float32")
        df["predictions_with_incentives"] = np.round(
            df["predictions_with_incentives"], 6
        )
        df["predictions_no_incentives"] = np.round(df["predictions_no_incentives"], 6)

        df.loc[df.dsp_code.isna(), "dsp_code"] = "NAN"
        df.loc[df.station_pair.isna(), "station_pair"] = "NAN"
        df.loc[df.dsp_type.isna(), "dsp_type"] = "NAN"

        df.loc[
            df["predictions_with_incentives"] > 150,
            "predictions_with_incentives",
        ] = 150
        df.loc[
            df["predictions_no_incentives"] > 150,
            "predictions_no_incentives",
        ] = 150

        df.loc[
            (df["predictions_with_incentives"] < 20)
            & (df["station_pair_tenure_0"] > 7),
            "predictions_with_incentives",
        ] = 20
#         df.loc[
#             (df["predictions_no_incentives"] < 20) & (df["station_pair_tenure_0"] > 7),
#             "predictions_no_incentives",
#         ] = 20
        return df

    def make_predictions(
        self,
        prediction_length,
        performance_data_path,
        model_features,
        model_path,
        scaler_path,
        transformer_path,
        local_predictions_path,
        local_predictions_path_no_input,
        local_coefficients_path,
        local_explanations_path,
        s3_predictions_upload_path,
        s3_predictions_upload_old_path,
        s3_predictions_upload_path_no_input,
        s3_coefficients_upload_path,
        s3_explanations_upload_path,
        dsp_data_path,
        input_data_path,
        s3_object_path_dict,
        run_date,
        s3_upload_flag,
        model_version,
    ):
        """
        This function is used to make predictions on performance input dataset using ModelTrainer Class

        """

        def make_predictions_no_incentives(performance_data, model_features):
            lower_performance_data = performance_data.copy()
            lower_performance_data["incentives_0"] = 0
            regression_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            transformer = joblib.load(transformer_path)
            lower_performance_data = (
                lower_performance_data[model_features].dropna().reset_index(drop=True)
            )
            X_infer_org = (
                lower_performance_data.drop(columns=["label"])
                .astype(float)
                .reset_index(drop=True)
            )
            X_infer = transformer.transform(X_infer_org)
            X_infer = scaler.transform(X_infer)

            X_infer = pd.DataFrame(X_infer, columns=X_infer_org.columns)

            y_infer = regression_model.predict(X_infer).astype("float64")
            return y_infer

        performance_data = pd.read_parquet(performance_data_path)
        
        steady_state_tenet_period = [
            i for i in range(5, 40) if i not in [25, 26, 27, 28, 29, 30]
        ]
        if (
            int(
                self.performance_stage_config_dict["prediction_year_week"][0].split(
                    "-"
                )[1]
            )
            in steady_state_tenet_period
        ):
            performance_data["active_da_count_0"] = np.minimum(performance_data["active_da_count_0"], 2*performance_data["completed_routes_max_3"])
            
        performance_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        performance_data = performance_data[
            ["dsp_type", "station_pair", "dsp_code"] + model_features
        ].dropna(subset=model_features)

        regression_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        transformer = joblib.load(transformer_path)

        high_performance_data = performance_data.copy()
        high_performance_data = (
            high_performance_data[model_features].dropna().reset_index(drop=True)
        )

        X_infer_org = (
            high_performance_data.drop(columns=["label"])
            .astype(float)
            .reset_index(drop=True)
        )
        X_infer_org[X_infer_org.ratio_station_pair_3_4 == np.inf].loc[
            :, "ratio_station_pair_3_4"
        ] = X_infer_org.ratio_station_pair_3_4.mean()

        X_infer = transformer.transform(X_infer_org)
        X_infer = scaler.transform(X_infer)

        X_explanations = self.make_explanations(regression_model, X_infer)

        X_explanations = pd.DataFrame(
            X_explanations.values, columns=X_infer_org.columns
        )
        X_explanations = pd.concat(
            [
                performance_data[["dsp_type", "station_pair", "dsp_code"]],
                X_explanations,
            ],
            axis=1,
        )

        X_infer = pd.DataFrame(X_infer, columns=X_infer_org.columns)
        X_infer = np.round(X_infer, 6)

        y_infer = regression_model.predict(X_infer).astype("float64")
        y_infer_no_incentives = make_predictions_no_incentives(
            performance_data, model_features
        )

        coef_df = pd.DataFrame()
        coef_df["feature"] = ["const"] + model_features[:-1]
        coef_df["coef"] = np.append(regression_model.intercept_, regression_model.coef_)

        performance_data.loc[:, "predictions_with_incentives"] = y_infer
        performance_data["predictions_no_incentives"] = y_infer_no_incentives

        performance_data = self.do_quality_checks(performance_data)

        performance_data = performance_data.reindex(
            columns=["dsp_type", "station_pair", "dsp_code"]
            + model_features
            + ["predictions_with_incentives", "predictions_no_incentives"]
        )

        performance_data = self.make_overrides(
            run_date, performance_data, input_data_path
        )
        
        if (
            int(
                self.performance_stage_config_dict["prediction_year_week"][0].split(
                    "-"
                )[1]
            )
            in steady_state_tenet_period
        ):
            performance_data["label"] = np.round(performance_data["completed_routes_2"] * performance_data["ratio_station_pair_3_4"], 6)
            performance_data["predictions_no_incentives"] = np.minimum(performance_data["predictions_with_incentives"], performance_data["label"])
            
        try:
            os.makedirs(
                local_predictions_path.split(
                    "/dsp_peak_scaling_linear_predictions.csv"
                )[0]
            )
            os.makedirs(
                local_coefficients_path.split(
                    "/dsp_peak_scaling_linear_coefficients.pqt"
                )[0]
            )
        except:
            logger.debug("Directory already exists")

        ## Add BIE relevant columns to the model output (provided by DSP Analytics)
        dsp_data = self.read_parquet_from_s3(dsp_data_path)
        performance_data.loc[:, "snapshot_date"] = dsp_data.stage_end.max()
        performance_data.loc[:, "publish_date"] = self.run_date
        performance_data.loc[:, "prediction_year"] = int(
            self.performance_stage_config_dict["prediction_year_week"][0].strip()[:4]
        )
        performance_data.loc[:, "prediction_week"] = int(
            self.performance_stage_config_dict["prediction_year_week"][0].strip()[-2:]
        )
        performance_data.loc[:, "prediction_year_week"] = str(
            self.performance_stage_config_dict["prediction_year_week"][0]
        )
        performance_data.loc[:, "model_version"] = model_version

        #         performance_data.rename(
        #             columns={
        #                 "DSP": "dsp_ind",
        #                 "Internal-Transfer": "internal_transfer_ind",
        #                 "Pinnacle": "pinnacle_ind",
        #                 "Pop-Up": "pop_up_ind",
        #             },
        #             inplace=True,
        #         )

        performance_data = self.merge_auxillary_data(
            performance_data, s3_object_path_dict, run_date
        )
        performance_data = self.make_corrections(
            performance_data, prediction_length, run_date
        )

        #         performance_data["station_pair"] = performance_data["station_pair"].astype('object')
        performance_data = performance_data[performance_data.station_pair.notna()]

        #         performance_data.loc[
        #             performance_data.dsp_code.isna(), "dsp_code"
        #         ] = performance_data.station_pair.apply(lambda x: x.split("-")[1])
        #         performance_data.rename(columns={"dsp_code": "dsp"}, inplace=True)

#         performance_data = performance_data[
#             [
#                 c
#                 for c in performance_data.columns
#                 if c not in ["lrp_station_vf_0", "dsp_quality_score_3"]
#             ]
#             + ["lrp_station_vf_0", "dsp_quality_score_3"]
#         ]  # sanity check
        
        performance_data = performance_data.astype({"launch_date":"datetime64[ns]", 
                                                    "transfer_in_date":"datetime64[ns]", 
                                                    "transfer_out_date":"datetime64[ns]",
                                                    "exit_date":"datetime64[ns]"})

        performance_data.to_csv(local_predictions_path, index=False)
        performance_data_parquet = performance_data.drop(columns=["publish_date", "prediction_year_week"])
        performance_data_no_input = performance_data[
            [
                "dsp_type",
                "station_pair",
                "dsp_code",
                "predictions_with_incentives",
                "predictions_no_incentives",
                "predictions_with_overrides",
                "snapshot_date",
                "prediction_year",
                "prediction_week",
                "model_version",
                "recruitment_type",
                "source_station",
                "destination_station",
                "launch_date",
                "transfer_in_date",
                "transfer_out_date",
                "exit_date",
            ]
        ]
        coef_df.to_parquet(local_coefficients_path, index=False)
        X_explanations.loc[:, "model_constant"] = coef_df[coef_df.feature == "const"][
            "coef"
        ][0]
        X_explanations.loc[:, "prediction_year_week"] = performance_data[
            "prediction_year_week"
        ]
        X_explanations.to_parquet(local_explanations_path, index=False)

        if s3_upload_flag:
            self.write_parquet_to_s3(
                performance_data_parquet,
                s3_predictions_upload_path,
            )
            self.write_parquet_to_s3(
                performance_data_no_input,
                s3_predictions_upload_path_no_input
            )
            self.upload_to_s3(
                local_download_path=local_predictions_path,
                s3_upload_path=s3_predictions_upload_old_path,
            )
            self.upload_to_s3(
                local_download_path=local_coefficients_path,
                s3_upload_path=s3_coefficients_upload_path,
            )
            self.upload_to_s3(
                local_download_path=local_explanations_path,
                s3_upload_path=s3_explanations_upload_path,
            )

        logger.debug(f"Inference Process completed, inference output generated with shape {performance_data.shape}")


def generate_stages(prediction_length, run_date, stage_type):
    """
    This function is used to generate stage for train and performance dataset

    """

    weeks = np.repeat(range(1, 53), 1)
    weeks = ["%02d" % x for x in weeks]
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    df_year_week = pd.DataFrame(
        [str(year) + "-" + str(week) for year in years for week in weeks],
        columns=["year_week"],
    )
    df_year_week = df_year_week.reset_index()
    df_year_week.rename(columns={"index": "time_period"}, inplace=True)

    if stage_type == "Train":
        run_date = run_date - timedelta(365)
        run_week = (run_date + timedelta(1)).isocalendar()[1]
        run_week = ["%02d" % x for x in [run_week]]
        run_year = run_date.year
        run_year_week = [
            str(year) + "-" + str(week) for year in [run_year] for week in run_week
        ]
    else:
        # run_date = run_date
        run_week = (run_date + timedelta(1)).isocalendar()[1]
        run_week = ["%02d" % x for x in [run_week]]
        run_year = run_date.year
        run_year_week = [
            str(year) + "-" + str(week) for year in [run_year] for week in run_week
        ]

    run_week_length = df_year_week[
        df_year_week.year_week == run_year_week[0]
    ].time_period.values[0]
    run_year_week = run_year_week[0]
    run_week = int(run_week[0])

    prediction_length = run_week_length + prediction_length

    if prediction_length > run_week_length + 4:
        stage1_length = run_week_length - 4
        stage2_length = run_week_length + 4
        stage3_length = prediction_length - 52
        stage4_length = stage2_length - 52
        stage5_length = stage1_length - 52

    else:
        stage1_length = run_week_length - 4
        stage2_length = prediction_length - 1
        stage3_length = prediction_length - 52
        stage4_length = stage2_length - 52
        stage5_length = stage1_length - 52

    prediction_year_week = df_year_week[
        df_year_week.time_period == prediction_length
    ].year_week.values.tolist()

    if run_week >= 33:
        stage1_prime_length = run_week_length - (run_week - 28)
        stage5_prime_length = stage1_prime_length - 52

        stage1_config = df_year_week[
            df_year_week.time_period == stage1_prime_length
        ].year_week.values.tolist()
        stage1_config.extend(
            df_year_week.loc[stage1_length : run_week_length - 1, "year_week"].values
        )

        stage2_config = df_year_week.loc[
            run_week_length + 1 : stage2_length, "year_week"
        ].values.tolist()

        stage3_config = df_year_week.loc[
            df_year_week.time_period == stage3_length, "year_week"
        ].values.tolist()

        stage4_config = df_year_week.loc[
            run_week_length + 1 - 52 : stage4_length, "year_week"
        ].values.tolist()

        stage5_config = df_year_week[
            df_year_week.time_period == stage5_prime_length
        ].year_week.values.tolist()
        stage5_config.extend(
            df_year_week.loc[stage5_length : run_week_length - 53, "year_week"].values
        )

    elif run_week <= 23:
        stage1_peak_length = run_week_length - (run_week + 2)
        stage5_peak_length = stage1_peak_length - 52

#         stage1_config = df_year_week[
#             df_year_week.time_period == stage1_peak_length
#         ].year_week.values.tolist()
        
        stage1_config = []
        stage1_config.extend(
            df_year_week.loc[stage1_length : run_week_length - 1, "year_week"].values
        )

        stage2_config = df_year_week.loc[
            run_week_length + 1 : stage2_length, "year_week"
        ].values.tolist()

        stage3_config = df_year_week.loc[
            df_year_week.time_period == stage3_length, "year_week"
        ].values.tolist()

        stage4_config = df_year_week.loc[
            run_week_length + 1 - 52 : stage4_length, "year_week"
        ].values.tolist()

        stage5_config = df_year_week[
            df_year_week.time_period == stage5_peak_length
        ].year_week.values.tolist()
        stage5_config.extend(
            df_year_week.loc[stage5_length : run_week_length - 53, "year_week"].values
        )

    else:
        stage1_config = df_year_week.loc[
            stage1_length : run_week_length - 1, "year_week"
        ].values.tolist()

        stage2_config = df_year_week.loc[
            run_week_length + 1 : stage2_length, "year_week"
        ].values.tolist()

        stage3_config = df_year_week.loc[
            df_year_week.time_period == stage3_length, "year_week"
        ].values.tolist()

        stage4_config = df_year_week.loc[
            run_week_length + 1 - 52 : stage4_length, "year_week"
        ].values.tolist()

        stage5_config = df_year_week.loc[
            stage5_length : run_week_length - 53, "year_week"
        ].values.tolist()

    stage_config_dict = {
        "run_year_week": run_year_week,
        "prediction_year_week": prediction_year_week,
        "stage1_config": stage1_config,
        "stage2_config": stage2_config,
        "stage3_config": stage3_config,
        "stage4_config": stage4_config,
        "stage5_config": stage5_config,
    }

    if stage_type != "Train":
        print(f" Model Run Year Week {[run_year_week]}")
        print(f" Prediction Year Week {prediction_year_week}")
        print(f" Stage 1 config {stage1_config}")
        print(f" Stage 2 config {stage2_config}")
        print(f" Stage 3 config {stage3_config}")
        print(f" Stage 4 config {stage4_config}")
        print(f" Stage 5 config {stage5_config}")

    return stage_config_dict


def main(
    prediction_length,
    run_date,
    mode,
    model_version,
    region,
    bucket,
    etl_prefix,
):
    """
    This function is to used run the model sequentially

    """

    s3_upload_flag = False
    if mode == "Prod":
        s3_upload_flag = True
    else:
        s3_upload_flag = False

    train_stage_config_dict = generate_stages(
        prediction_length, run_date, stage_type="Train"
    )
    performance_stage_config_dict = generate_stages(
        prediction_length, run_date, stage_type="Performance"
    )
    prediction_week = performance_stage_config_dict["prediction_year_week"][0]

    if prediction_length <= 0:
        raise Exception(
            "Prediction length less than 0 is not allowed, please enter a value greater than 0"
        )
    else:
        # s3 download object paths (s3 versions stay in merge update)
        s3_object_path_dict = {
            "agd_path": f"{etl_prefix}/dsp_peak_scaling_agd.csv000",
            "attrition_path": f"{etl_prefix}/dsp_peak_scaling_attrition.csv000",
            "boc_path": f"{etl_prefix}/dsp_peak_scaling_boc.csv000",
            "completed_routes_path": f"{etl_prefix}/dsp_peak_scaling_completed_routes.csv000",
            "grounded_path": f"{etl_prefix}/dsp_peak_scaling_grounded.csv000",
            "incentives_path": f"{etl_prefix}/dsp_peak_scaling_incentives.csv000",
            "nh_path": "etl_files/dsp_peak_scaling_nh.csv000",
            "nh_new_path": f"{etl_prefix}/dsp_peak_scaling_nh_new.csv000",
            "robl_path": f"{etl_prefix}/dsp_peak_scaling_robl.csv000",
            "scorecard_path": f"{etl_prefix}/dsp_peak_scaling_scorecard.csv000",
            "vin_path": f"{etl_prefix}/dsp_peak_scaling_vin.csv000",
            "mtp_path": f"{etl_prefix}/dsp_peak_scaling_mtp_lpt.csv000",
            "transfers_path": f"{etl_prefix}/dsp_peak_scaling_transfers.csv000",
            "exits_path": f"{etl_prefix}/dsp_peak_scaling_exits.csv000",
            "station_region_path": f"{etl_prefix}/dsp_peak_scaling_station_region_data.csv000",
            "da_hiring_signal_path": f"{etl_prefix}/dsp_peak_scaling_da_hiring_signal.csv000",
            "stp_path": f"{etl_prefix}/dsp_peak_scaling_stp_data.csv000",
            "station_volume_forecast_path": f"{etl_prefix}/dsp_peak_scaling_volume_forecast.csv000",
            "station_volume_actuals_path": f"{etl_prefix}/dsp_peak_scaling_station_volume_actuals.csv000",
            "dsp_quality_path": f"{etl_prefix}/dsp_peak_scaling_quality_score.csv000",
            "lrp_sp_universe_path": "etl_files/dsp_peak_scaling_lrp_sp_universe.csv000",
            "br_path": f"{etl_prefix}/dsp_peak_scaling_business_review.csv000",
            "lrp_hiring_signal_path": f"{etl_prefix}/dsp_peak_scaling_lrp_hiring_signal.csv000",
        }

    # Input Files Local Path
    dsp_data_local_path = os.path.join(
        os.getcwd(), f"input_files/{run_date}/dsp_peak_scaling_dsp_data.pqt"
    )
    input_data_local_path = os.path.join(
        os.getcwd(), f"input_files/{run_date}/dsp_peak_scaling_input_data.pqt"
    )

    # Input Files S3 Path
    s3_dsp_data_upload_path = f"{etl_prefix}/dsp_data.pqt"
    s3_input_upload_path = f"input_files/publish_date={run_date}/input_data.pqt"

    univariate_train_predictions_local_file_path = os.path.join(
        os.getcwd(), "dsp_peak_scaling_univariate_predictions_train.csv"
    )
    univariate_train_model_local_file_path = os.path.join(
        os.getcwd(), "dsp_peak_scaling_univariate_train_model.pkl"
    )

    univariate_performance_predictions_local_file_path = os.path.join(
        os.getcwd(), "dsp_peak_scaling_univariate_performance.csv"
    )
    univariate_performance_model_local_path = os.path.join(
        os.getcwd(), "dsp_peak_scaling_univariate_train_model.pkl"
    )

    # ModelTrainer Object Artifacts: local/s3 upload paths
    local_power_transformer_path = os.path.join(
        os.getcwd(),
        f"model_artifacts/model_files/{run_date}/{prediction_week}/dsp_peak_scaling_power_transformer.pkl",
    )
    local_scaler_path = os.path.join(
        os.getcwd(),
        f"model_artifacts/model_files/{run_date}/{prediction_week}/dsp_peak_scaling_linear_scaler.pkl",
    )
    trained_file_path = os.path.join(
        os.getcwd(),
        f"model_artifacts/model_files/{run_date}/{prediction_week}/dsp_peak_scaling_trained.csv",
    )
    test_file_path = os.path.join(
        os.getcwd(),
        f"model_artifacts/model_files/{run_date}/{prediction_week}/dsp_peak_scaling_test.csv",
    )
    linear_model_path = os.path.join(
        os.getcwd(),
        f"model_artifacts/model_files/{run_date}/{prediction_week}/dsp_peak_scaling_linear_model.pkl",
    )
    local_model_performance_path = os.path.join(
        os.getcwd(),
        f"model_artifacts/model_files/{run_date}/{prediction_week}/dsp_peak_scaling_model_performance.csv",
    )

    # InferenceHandler Object Artifacts: local/s3 upload paths
    linear_predictions_path = os.path.join(
        os.getcwd(),
        f"model_artifacts/output/{run_date}/{prediction_week}/dsp_peak_scaling_linear_predictions.csv",
    )
    linear_predictions_path_no_input = os.path.join(
        os.getcwd(),
        f"model_artifacts/output/{run_date}/{prediction_week}/dsp_peak_scaling_linear_predictions_no_input.csv",
    )
    linear_coeffcients_path = os.path.join(
        os.getcwd(),
        f"model_artifacts/output/{run_date}/{prediction_week}/dsp_peak_scaling_linear_coefficients.pqt",
    )
    linear_explanations_path = os.path.join(
        os.getcwd(),
        f"model_artifacts/output/{run_date}/{prediction_week}/dsp_peak_scaling_linear_explanations.pqt",
    )
    s3_univariate_train_model_upload_path = (
        f"trained_model_files/{run_date}/dsp_peak_scaling_univariate_model_train.pkl"
    )
    s3_univariate_train_predictions_upload_path = (
        f"inferences/{run_date}/dsp_peak_scaling_univariate_predictions_train.csv"
    )
    s3_univariate_performance_model_upload_path = (
        f"trained_model_files/{run_date}/dsp_peak_scaling_univariate_model_train.pkl"
    )
    s3_univariate_performance_predictions_upload_path = (
        f"inferences/{run_date}/dsp_peak_scaling_univariate_predictions_train.csv"
    )

    # Trained Files Local Path
    train_input_local_path = os.path.join(
        os.getcwd(), f"input_files/{run_date}/dsp_peak_scaling_train_input.pqt"
    )
    performance_input_local_path = os.path.join(
        os.getcwd(),
        f"input_files/{run_date}/dsp_peak_scaling_performance_input.pqt",
    )

    # Production s3 paths: model artifacts
    s3_train_input_upload_path = (
        f"trained_model_files/publish_date={run_date}/train_input_data.pqt"
    )
    s3_performance_input_upload_path = (
        f"trained_model_files/publish_date={run_date}/performance_input_data.pqt"
    )
    s3_power_transformer_upload_path = (
        f"trained_model_files/publish_date={run_date}/dsp_peak_scaling_transformer.pkl"
    )
    s3_scaler_upload_path = f"trained_model_files/publish_date={run_date}/dsp_peak_scaling_linear_scaler.pkl"
    s3_linearmodel_upload_path = (
        f"trained_model_files/publish_date={run_date}/dsp_peak_scaling_linear_model.pkl"
    )

    s3_linear_coefficients_upload_path = f"model_coefficients/publish_date={run_date}/region={region}/prediction_year_week={prediction_week}/dsp_peak_scaling_linear_coefficients.pqt"
    s3_linear_explanations_upload_path = f"model_explanations/publish_date={run_date}/region={region}/prediction_year_week={prediction_week}/dsp_peak_scaling_linear_explanations.pqt"
    s3_linear_predictions_upload_path = f"inferences_v4/publish_date={run_date}/region={region}/prediction_year_week={prediction_week}/dsp_peak_scaling_linear_predictions.pqt"
    s3_predictions_upload_old_path = f"inferences/{run_date}/{prediction_week}/dsp_peak_scaling_linear_predictions.csv"
    s3_linear_predictions_upload_path_no_input = f"inferences_no_input/publish_date={run_date}/region={region}/prediction_year_week={prediction_week}/dsp_peak_scaling_linear_predictions_no_input.pqt"

    ID_list = [
        "year",
        "week",
        "year_week",
        "country_code",
        "program_type",
        "station_pair",
        "station_code",
        "dsp_code",
        "region",
    ]

    feature_list = [
        "attr_rate",
        "active_da_count",
        "adaccuracy_metric",
        "agg_dcr_metric",
        "box_truck_route_share",
        "business_constrained_metric",
        "cas_metric",
        "category_compliancesafety_metric",
        "category_quality_metric",
        "category_team_metric",
        "cc_metric",
        "cdv_route_share",
        "completed_routes",
        "completed_routes_max",
        "compliance_metric",
        "customerescalation_metric",
        "da_experience",
        "da_tenure",
        "dar_metric",
        "dcr_metric",
        "delivered",
        "dispatched",
        "driver_affinity",
        "dsp_final_metric",
        "dsp_quality_score",
        "dsp_tenure_week",
        "station_pair_tenure",
        "dsp_peak_years",
        "dsp_nhscore",
        "dsp_nhscore_rank",
        "station_pair_peak_years",
        "exertion",
        "extra_large_van_route_share",
        "final_route_target",
        "final_route_target_max",
        "grounded_vans",
        "grounded_days",
        "in_station_delivery",
        "incentives",
        "t6_isight_points",
        "t6_cr_isight_points",
        "large_van_route_share",
        "late_cancelled_by_dsp",
        "late_cancelled_by_dsp_max",
        "negative_feedback_metric",
        "nr_routes",
        "nursery_route_share",
        "ov_pkg",
        "pkg_agd_mean",
        "plan_on_road_minutes",
        "plan_on_road_minutes_nr",
        "plan_on_road_minutes_sp",
        "pod_metric",
        "rank_reliability_within_station",
        "reliability",
        "requested_routes",
        "requested_routes_max",
        "rivian_route_share",
        "robl_30",
        "robl_30_nr",
        "robl_30_sp",
        "robl_90",
        "robl_mins",
        "robl_mins_nr",
        "robl_mins_sp",
        "route_adherence",
        "route_stop_agd_sum",
        "safety_metric",
        "seatbelt_off_metric",
        "spr",
        "step_van_route_share",
        "sustained_das",
        "sustained_high_das",
        "sustained_poor_das",
        "swc_metric",
        "tech_pva",
        "tech_pva_nr",
        "tech_pva_sp",
        "vin_active_rate",
        "vin_branded",
        "vin_branded_rate",
        "vin_total",
        "vol_share_cycle1",
        "weekly_routes",
        "weekly_work_days",
        "wes",
        "whc_metric",
        # "dsp_type",
    ]

    model_features = [
        "DSP 2.0",
        "DSP 1.0",
        "Walker",
        "Internal-Transfer",
        "Pinnacle",
        "Pop-Up",
        "attr_rate_3",
        "active_da_count_0",
        "completed_routes_max_max_1",
        "completed_routes_2",
        "completed_routes_max_3",
        "da_experience_1",
        "distance",
        "driver_affinity_1",
        "dsp_final_metric_1",
        "dsp_final_metric_3",
        "dsp_nhscore_1",
        "dsp_nhscore_rank_1",
        "dsp_quality_score_1",
        "dsp_tenure_week_0",
        "final_route_target_2",
        "incentives_0",
        "late_cancelled_by_dsp_3",
#         "lrp_station_vf_0",
        "ratio_station_pair_to_station_1",
        "region_3_5_routes_diff",
        "ratio_station_pair_3_4",
        "reliability_1",
        "reliability_3",
        "requested_routes_max_1",
        "robl_mins_1",
        "route_stop_agd_sum_1",
        "spr_3",
        "station_3_5_routes_diff",
        "station_pair_tenure_0",
        "sustained_high_das_1",
        "t6_cr_isight_points_1",
        "t6_isight_points_1",
        "vin_active_rate_1",
        "vin_active_rate_2",
        "vin_active_rate_3",
        "vin_branded_rate_1",
        "vin_branded_rate_3",
        "vin_total_1",
        "vin_total_3",
        "vol_share_cycle1_1",
        "vol_share_cycle1_3",
        "label",
    ]

    # Feature Engineering
    feature_handler = FeatureHandler(
        input_data_path=s3_input_upload_path,
        train_stage_config_dict=train_stage_config_dict,
        performance_stage_config_dict=performance_stage_config_dict,
        prediction_length=prediction_length,
        s3_object_path_dict=s3_object_path_dict,
        bucket=bucket,
    )

    feature_handler.create_training_input(
        train_stage_config_dict=train_stage_config_dict,
        feature_list=feature_list,
        run_date=run_date,
        univariate_local_file_path=univariate_train_predictions_local_file_path,
        local_download_path=train_input_local_path,
        s3_upload_path=s3_train_input_upload_path,
        s3_upload_flag=s3_upload_flag,
    )

    df_performance = feature_handler.stage_performance_input(
        train_input_path=train_input_local_path,
        performance_stage_config_dict=performance_stage_config_dict,
        feature_list=feature_list,
    )

    feature_handler.create_performance_input(
        valid_input=df_performance,
        s3_dsp_data_upload_path=s3_dsp_data_upload_path,
        performance_stage_config_dict=performance_stage_config_dict,
        local_download_path=performance_input_local_path,
        run_date=run_date,
        prediction_length=prediction_length,
        s3_upload_path=s3_performance_input_upload_path,
        s3_upload_flag=s3_upload_flag,
        model_version=model_version,
    )

    ## Model Training ==============================
    estimator = ModelTrainer(bucket)

    estimator.do_train_test_transformation_split(
        train_data_path=train_input_local_path,
        model_features=model_features,
        local_power_transformer_path=local_power_transformer_path,
        local_scaler_path=local_scaler_path,
        s3_power_transformer_path=s3_power_transformer_upload_path,
        s3_scaler_upload_path=s3_scaler_upload_path,
        s3_upload_flag=s3_upload_flag,
    )
    estimator.do_model_training(
        train_data_path=train_input_local_path,
        local_model_path=linear_model_path,
        local_train_file_path=trained_file_path,
        local_test_file_path=test_file_path,
        local_model_performance_path=local_model_performance_path,
        s3_model_upload_path=s3_linearmodel_upload_path,
        s3_upload_flag=s3_upload_flag,
    )

    ## Model Inference ==============================
    linear_predictor = InferenceHandler(
        run_date=run_date,
        s3_path_dict=s3_object_path_dict,
        performance_stage_config_dict=performance_stage_config_dict,
        bucket=bucket,
    )
    linear_predictor.make_predictions(
        prediction_length=prediction_length,
        performance_data_path=performance_input_local_path,
        model_features=model_features,
        model_path=linear_model_path,
        scaler_path=local_scaler_path,
        transformer_path=local_power_transformer_path,
        local_predictions_path=linear_predictions_path,
        local_predictions_path_no_input=linear_predictions_path_no_input,
        local_coefficients_path=linear_coeffcients_path,
        local_explanations_path=linear_explanations_path,
        s3_predictions_upload_path=s3_linear_predictions_upload_path,
        s3_predictions_upload_path_no_input=s3_linear_predictions_upload_path_no_input,
        s3_coefficients_upload_path=s3_linear_coefficients_upload_path,
        s3_explanations_upload_path=s3_linear_explanations_upload_path,
        s3_predictions_upload_old_path=s3_predictions_upload_old_path,
        dsp_data_path=s3_dsp_data_upload_path,
        input_data_path=s3_input_upload_path,
        s3_object_path_dict=s3_object_path_dict,
        run_date=run_date,
        s3_upload_flag=s3_upload_flag,
        model_version=model_version,
    )

    logger.debug(
        f"DSP Peak Scaling Model Code run completed for stage {prediction_week}"
    )
    
class UnitTester(DataHandler):
    def __init__(
        self, 
        run_date, 
        model_version,
        bucket=None
    ):
        DataHandler.__init__(self, bucket)
        self.run_date = run_date
        self.model_version = model_version
        self.bucket = bucket
        
    def do_wape_calculations(self, df):
        df["absolute_diff_with_incentives"] = np.where(
            df.predictions_with_incentives < df.completed_routes_target,
            np.abs(df.predictions_with_incentives - df.completed_routes_target),
            np.where(
                df.predictions_with_incentives < df.final_route_target, 
                np.abs(df.predictions_with_incentives - df.completed_routes_target),
                np.abs(df.completed_routes_target - df.final_route_target)
            )
        )

        df["absolute_diff_no_incentives"] = np.where(
            df.predictions_no_incentives < df.completed_routes_target,
            np.abs(df.predictions_no_incentives - df.completed_routes_target),
            np.where(
                df.predictions_no_incentives < df.final_route_target, 
                np.abs(df.predictions_no_incentives - df.completed_routes_target),
                np.abs(df.completed_routes_target - df.final_route_target)
            )
        )

        df["absolute_diff_with_overrides"] = np.where(
            df.predictions_with_overrides < df.completed_routes_target,
            np.abs(df.predictions_with_overrides - df.completed_routes_target),
            np.where(
              df.predictions_with_overrides < df.final_route_target, 
              np.abs(df.predictions_with_overrides - df.completed_routes_target),
              np.abs(df.completed_routes_target - df.final_route_target)
            )
        )
        df = df.groupby(["year", "week"]).agg(
            {
                "absolute_diff_with_incentives":"sum", 
                "absolute_diff_no_incentives":"sum", 
                "absolute_diff_with_overrides":"sum", 
                "completed_routes_target":"sum"
            }
        ).reset_index()
        df["wape_with_incentives"] = np.round(100 * df["absolute_diff_with_incentives"] / df["completed_routes_target"], 2)
        df["wape_no_incentives"] = np.round(100 * df["absolute_diff_no_incentives"] / df["completed_routes_target"], 2)
        df["wape_with_overrides"] = np.round(100 * df["absolute_diff_with_overrides"] / df["completed_routes_target"], 2)
        return df
        
    def run_model_inference_prior_peak(self, model_version):
        """
        This function is used to run the current model version for last year's peak period and generate output to calculate WAPE
        
        """
        peak_publish_date = datetime(2023, 9, 20).date()
        peak_prediction_length = 12
        mode = "Prod"
        model_version = model_version
        bucket = "dsp-capacity-forecast"
        etl_prefix_ut = f"etl_files/{peak_publish_date - timedelta(2)}"
        region = "NA"
        
        main(peak_prediction_length, 
              peak_publish_date, 
              mode, 
              model_version, 
              region, 
              bucket,
              etl_prefix_ut)
    
    def run_wape_tester(self, region, etl_prefix):
        peak_publish_date = datetime(2023, 9, 20).date()
        key = self.read_s3_keys_from_prefix(f"inferences_v4/publish_date={peak_publish_date}/region={region}/prediction_year_week=2023-50")
        df_predictions = self.read_parquet_from_s3(key[0])
        df_actuals = self.read_csv_from_s3(f"{etl_prefix}/dsp_peak_scaling_completed_routes.csv000")
        df_actuals = df_actuals[(df_actuals.year==2023) & (df_actuals.week==50)].reset_index(drop=True)
        df_actuals = df_actuals[["station_pair", "year", "week", "completed_routes_target", "final_route_target"]]
        report = df_predictions.merge(df_actuals, left_on=["station_pair", "prediction_year", "prediction_week"], right_on=["station_pair", "year", "week"], how="inner")
        wape = self.do_wape_calculations(report)
        logger.debug(f"Weighted Absolute Percentage Error for week 50 last year = {wape['wape_with_incentives']}")
        
    def run_unit_tester(self, region, etl_prefix):
        self.run_wape_tester(region, etl_prefix)
        self.run_model_inference_prior_peak(self.model_version)
        self.run_wape_tester(region, etl_prefix)


if __name__ == "__main__":
    run_id = sys.argv[1]  # unique run id assigned by the batch job
    run_date = sys.argv[2]
    prediction_length = int(sys.argv[3])
    mode = sys.argv[4]
    model_version = sys.argv[5]
    region = sys.argv[6]
    bucket = sys.argv[7]
    etl_prefix = sys.argv[8]

    run_date_format = "%Y-%m-%d"
    run_date = datetime.strptime(run_date, run_date_format).date()

    # stage3_config tunable using the lambda function that triggers the batch job
    main(
        prediction_length,
        run_date,
        mode,
        model_version,
        region,
        bucket,
        etl_prefix,
    )
    
