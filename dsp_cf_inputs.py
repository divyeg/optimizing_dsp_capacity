#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import re
import subprocess

print(f"Python version used by batch {sys.version}")
print(f"System Executable {sys.executable}")


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
# install("awswrangler")
# install("scikit-learn==1.2")
# uninstall("typing_extensions")
# upgrade("typing_extensions==4.7.1")
# install("numpy==1.23.1")

import pandas as pd
import numpy as np
import time
import boto3
import json
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from multiprocessing import Pool
from functools import partial
from uuid import uuid4
import io
from io import StringIO

from loguru import logger

import psycopg2

# import rs_utility as utl
import awswrangler as wr
from sklearn.impute import KNNImputer

np.random.seed(6)
os.environ["PYTHONHASHSEED"] = str(6)

# removed_features = ['attr_rate_y0_2']


class DataHandler(object):
    def __init__(self, bucket):
        self.bucket = bucket

    def get_conn(self, secret_name, cluster_name):
        """
        The function is used to run queries on AMZL Analytics Redshift Cluster using RJDBC driver

        Params:
        -----------------------
        cluster_name = redshift cluster name (choose from amzlanalytics, amzl-bia-compute)

        Returns:
        -----------------------
        connection = connection object to run queries on the Redshift Cluster

        """
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
        bucket = s3 bucket used to download and upload objects
        download_object_path = object prefix in s3 buckets for download action
        download_local_path = object path in local directory for storing in local action

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


class PreprocessHandler(DataHandler):
    def __init__(
        self,
        s3_object_path_dict,
        stage_config_dict,
        run_date,
        bucket=None,
    ):
        self.stage_config_dict = (
            stage_config_dict  # we are using performance stage config dict
        )
        self.s3_object_path_dict = s3_object_path_dict
        self.stage1_config = stage_config_dict["stage1_config"]
        self.stage2_config = stage_config_dict["stage2_config"]
        self.stage3_config = stage_config_dict["stage3_config"]
        self.stage4_config = stage_config_dict["stage4_config"]
        self.stage5_config = stage_config_dict["stage5_config"]
        self.run_date = run_date
        self.input_data = pd.DataFrame()
        self.dsp_data = pd.DataFrame()

        DataHandler.__init__(self, bucket)

    def timer(f):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = f(*args, **kwargs)
            stop_time = time.time()
            dt = np.round(stop_time - start_time, 7)
            print(f"Time taken to run the {f} = {dt} seconds")
            return result

        return wrapper

    def do_preprocessing_dsp_data(self):
        """
        This function is used to perform preprocessing steps on dsp data

        Params:
        ----------------------------------------------------------------
        path: dsp data file path

        Returns:
        -------------------------
        df: pd.DataFrame()
                        preprocessed dsp data called in create_input function

        """
        df = self.dsp_data.copy()
        df = df.loc[df.year >= 2020]
        df = df.drop(columns=["dsp_type"])
        df = df.astype(
            {
                "stage_end": "datetime64[ns]",
                "stage_start": "datetime64[ns]",
            }
        )
        df["week"] = ["%02d" % x for x in df.week]
        df.loc[:, "year_week"] = df["year"].astype(str) + "-" + df["week"].astype(str)

        df = df.drop(
            columns=[
                "dvcr_metric",
                "safe_driving_metric",
            ]
        )

        numeric_columns = df.select_dtypes(include="number").columns.tolist()[2:]
        df = (
            df.groupby(
                [
                    "year",
                    "week",
                    "year_week",
                    "station_pair",
                    "station_code",
                    "dsp_code",
                    "country_code",
                    "program_type",
                ]
            )[numeric_columns]
            .median()
            .sort_values(by=["station_pair", "year", "week"])
            .reset_index()
        )

        station_region = self.read_csv_from_s3(
            self.s3_object_path_dict["station_region_path"]
        )
        df = df.merge(
            station_region,
            left_on=["station_code", "country_code"],
            right_on=["location_id", "country_code"],
            how="left",
        ).iloc[:, :-1]

        if df.shape[0] < 600:
            raise Exception("DSP Data not sufficient, has less than 600 rows")
        else:
            self.dsp_data = df.copy()
            logger.debug(f"Preprocssing DSP Data is complete.")

        return df

    def do_preprocessing_vin_data(self):
        """
        This function is used to perform preprocessing steps on vin data

        Params:
        ----------------------------------------------------------------

        Returns:
        -------------------------
        df: pd.DataFrame()
                        preprocessed vin data called in create_input function

        """
        df = self.read_csv_from_s3(self.s3_object_path_dict["vin_path"])
        df = df.loc[df.year >= 2020]
        df["station_pair"] = df["station_code"] + "-" + df["dsp_code"]
        df["week"] = ["%02d" % x for x in df.week]

        # removing DSPs with multiple dsp_types (edge cases)
        df_temp = df.groupby(["station_pair"]).dsp_type.nunique().reset_index()
        df_temp = df_temp[df_temp.dsp_type > 1]
        df = df[~(df.station_pair.isin(df_temp.station_pair))].reset_index(drop=True)

        df = (
            df.groupby(
                [
                    "station_code",
                    "dsp_code",
                    "station_pair",
                    "country_code",
                    "program_type",
                    "dsp_type",
                    "year",
                    "week",
                ]
            )[
                [
                    "vin_branded",
                    "vin_active",
                    "vin_utilized",
                    "vin_total",
                    "vin_model_year",
                ]
            ]
            .median()
            .sort_values(by=["station_pair", "year", "week"])
            .reset_index()
        )
        df["vin_branded_rate"] = df["vin_branded"] / df["vin_total"]
        df["vin_active_rate"] = df["vin_active"] / df["vin_total"]
        df["vin_utilized_rate"] = df["vin_utilized"] / df["vin_total"]
        df.fillna(0, inplace=True)

        if df.shape[0] < 200:
            raise Exception("VIN Data not sufficient, has less than 200 rows")
        else:
            logger.debug(f"Preprocessing Vin Data is complete.")

        return df

    def do_preprocessing_attrition_data(self):
        """
        This function is used to preprocess attrition data

        Params:
        ----------------------------------------------------------------

        Returns:
        -------------------------
        df: pd.DataFrame()
                        preprocessed attrition data called in create_input function

        """

        df = self.read_csv_from_s3(self.s3_object_path_dict["attrition_path"])
        df = df.loc[df.year >= 2020]
        df.loc[:, "week"] = df.week.astype(str).str[-2:]
        df.drop(columns=["program_type"], inplace=True)
        df["dsp_type"] = np.where(
            df.dsp_type.isin(["1.0", "2.0"]), "DSP " + df["dsp_type"], df.dsp_type
        )

        # removing DSPs with multiple dsp_types (edge cases)
        df_temp = df.groupby(["station_pair"]).dsp_type.nunique().reset_index()
        df_temp = df_temp[df_temp.dsp_type > 1]
        df = df[~(df.station_pair.isin(df_temp.station_pair))].reset_index(drop=True)

        df = (
            df.groupby(
                [
                    "station_pair",
                    "station_code",
                    "dsp_code",
                    "country_code",
                    # "program_type",
                    "year",
                    "week",
                    "dsp_type",
                ]
            )[["attr_da_count", "active_da_count"]]
            .sum()
            .reset_index()
        )
        df.loc[:, "attr_rate"] = df["attr_da_count"] / df["active_da_count"]

        df.fillna(0, inplace=True)

        if df.shape[0] < 200:
            raise Exception("Attrition Data has less than 200 rows")
        else:
            logger.debug(f"Preprocessing Attrition Data is complete")
        return df

    def do_preprocessing_nh_data(self):
        """
        This function is used to preprocess the nh data

        Params:
        ----------------------------------------------------------------
        None

        Returns:
        -------------------------
        df: pd.DataFrame()
                        preprocessed network health data called in create_input function

        """

        df = self.read_csv_from_s3(self.s3_object_path_dict["nh_path"])
        df["year"] = 2022
        df.loc[:, "year_week"] = max(self.stage5_config)
        df = df[
            [
                "dsp",
                "station",
                "year_week",
                "country",
                "dsp_nhscore",
                "dsp_nhscore_rank",
                "station_nhscore",
                "station_rank",
            ]
        ].copy()
        df.rename(
            columns={"station_rank": "station_nhscore_rank", "country": "country_code"},
            inplace=True,
        )

        df_new = self.read_csv_from_s3(self.s3_object_path_dict["nh_new_path"])

        df_new.loc[:, "year_week"] = (
            df_new["year"].astype(str) + "-" + df_new["week"].astype(str)
        )
        df_new = df_new.drop(columns=["week", "year", "program_type"])
        df_new["dsp_type"] = np.where(
            df_new.dsp_type.isin(["1.0", "2.0"]),
            "DSP " + df_new["dsp_type"],
            df_new.dsp_type,
        )

        df = pd.concat([df, df_new], ignore_index=True)
        df["station_pair"] = df["station"] + "-" + df["dsp"]
        df["dsp_nhscore_rank"] = df["dsp_nhscore_rank"].astype(float)
        df.rename(
            columns={
                "station_rank": "station_nhscore_rank",
                "dsp": "dsp_code",
                "station": "station_code",
            },
            inplace=True,
        )

        if df.shape[0] < 200:
            raise Exception("Network Health Data Incomplete, has less than 200 rows")
        else:
            logger.debug(f"Preprocessing Network Health Data is complete.")

        return df

    def do_preprocessing_incentive_data(self):
        """
        This function is used to preprocess the network health data

        Params:
        ----------------------------------------------------------------
        None

        Returns:
        -------------------------
        df: pd.DataFrame()
                        preprocessed incentive data called in create_input function

        """
        df = self.read_csv_from_s3(self.s3_object_path_dict["incentives_path"])
        df["week"] = ["%02d" % x for x in df.week]
        # df["station_pair"] = df["station"] + "-" + df["provider"]
        df.fillna(0, inplace=True)

        if df.shape[0] < 200:
            raise Exception("NH Data has less than 200 rows")
        else:
            logger.debug(f"Preprocessing Incentives Data is complete.")

        return df

    def merge_dsp_quality_score(self, input_data):
        """
        This function is used to preprocess the dsp quality score data

        Params:
        ---------------------------
        None

        Returns:
        df: pd.DataFrame()
                        preprocessed dsp quality score data in create_input_function

        """
        df = self.read_csv_from_s3(self.s3_object_path_dict["dsp_quality_path"])
        df.loc[:, "dsp_quality_score"] = (
            df["quality_score_num"] / df["quality_score_denom"]
        )
        df = df[["dsp_code", "country_code", "year_week", "dsp_quality_score"]]

        if df.shape[0] < 200:
            raise Exception("DSP Quality Data has less than 200 rows")
        else:
            logger.debug(f"Preprocessing DSP Quality Data is complete")

        input_data = input_data.merge(
            df, on=["dsp_code", "country_code", "year_week"], how="left"
        )
        return input_data

    def do_download_from_s3(self, s3_object_path_dict):
        """
        This function is used to download data from s3 bucket

        Params:
        ----------------------------------------------------------------
        s3_object_path_dict: dict
                        dictionary containing s3 locations of inputs files

        Returns:
        -------------------------
        None

        """

        try:
            os.makedirs(
                self.local_path_dict["agd_path"].split("/dsp_peak_scaling_agd.csv")[0]
            )
        except:
            logger.debug("Directory already exists")

        for key in self.local_path_dict.keys():
            self.download_from_s3(s3_object_path_dict[key], self.local_path_dict[key])

        logger.debug(f"Download from s3 operation completed successfully")

    def create_dsp_data(self, local_file_path, s3_upload_path, s3_upload_flag):
        """
        The function is used to create the dsp data by stiching together multiple data sources from redshift tables

        Params:
        ----------------------------------------------------------------
        local_file_path: string
                        local file path to store dsp data
        s3_upload_path: string
                        s3 object path to store dsp data
        s3_upload_flag: boolean
                        boolean flag to indicate if file needs to be uplaoded to s3 bucket

        Returns:
        -------------------------
        None

        """
        first_run = 0
        for metric in [
            "completed_routes_path",
            "agd_path",
            "grounded_path",
            "robl_path",
            "scorecard_path",
            "boc_path",
        ]:
            temp = self.read_csv_from_s3(self.s3_object_path_dict[metric])
            # the following code is added to debug the dataset start date issue
            temp["year_week"] = (
                temp["year"].astype(str) + "-" + temp["week"].astype(str)
            )
            logger.debug(f"Starting week of {metric} dataset is {temp.year_week.min()}")
            temp = temp.drop(columns="year_week")

            if first_run == 0:
                data = temp.copy()
                first_run = 1
            else:
                data = data.merge(
                    temp,
                    on=[
                        "station_pair",
                        "station_code",
                        "dsp_code",
                        "country_code",
                        "program_type",
                        "year",
                        "week",
                    ],
                    how="left",
                )
        data.rename(
            columns={
                "dsp_tenure_x": "dsp_tenure_week",
                "dsp_tenure_y": "dsp_tenure_day",
            },
            inplace=True,
        )

        if data.shape[0] < 600:
            raise Exception("DSP Data has less than 200 rows")
        else:
            self.dsp_data = data.copy()

        try:
            os.makedirs(local_file_path)
        except:
            logger.debug("Directory already exists")

        data.to_parquet(
            os.path.join(local_file_path, "dsp_peak_scaling_dsp_data.pqt"), index=False
        )

        if s3_upload_flag:
            self.upload_to_s3(
                os.path.join(local_file_path, "dsp_peak_scaling_dsp_data.pqt"),
                s3_upload_path,
            )
        else:
            logger.debug(f"dsp data not uploaded to s3 - working in development mode")

        logger.debug(f"Action - Creating DSP Data is complete")
        print()

    def find_similar_dsps(self, df):
        """
        This function return the list of dataframes divided by regions to ensure similar DSPs within same regions only are learning from each other

        Params:
        ----------------------------------------------------------------
        df: pd.DataFrame()
                        input dataset required to create sub-dataframes for each region

        Returns:
        -------------------------
        None

        """
        group_list = []
        for group, df_group in df.groupby("region"):
            group_list.append(df_group)

        return group_list

    def do_similarity_based_imputations(
        self, df, feature_list, performance_stage_config_dict, ID_list
    ):
        """
        This function is used to make missing value imputations for similar DSPs using Nearest Neighbours
        Params:
        ----------------------------------------------------------------
        df: pd.DataFrame()
                        input dataset required to impute missing values using KNN algorithm
        feature_list: list
                        list of features to be used for imputations
        performance_stage_config_dict: dict
                        dictionary containing performance stage configuration parameters
        ID_list: list
                        list of identifiers used for imputation of missing values

        Returns:
        -------------------------
        df: pd.DataFrame()
                        input dataset with missing values imputed using KNN algorithm

        """
        weeks = np.repeat(range(1, 53), 1)
        weeks = ["%02d" % x for x in weeks]
        years = [2020, 2021, 2022, 2023, 2024, 2025]
        df_year_week = pd.DataFrame(
            [str(year) + "-" + str(week) for year in years for week in weeks],
            columns=["year_week"],
        )
        df_year_week["year"] = df_year_week["year_week"].apply(lambda x: int(x[:4]))
        df_year_week["week"] = df_year_week["year_week"].apply(lambda x: x[5:])
        df_join_dates = df_year_week[
            (df_year_week["year_week"] >= df["year_week"].min())
            & (
                df_year_week["year_week"]
                < performance_stage_config_dict["run_year_week"]
            )
        ]
        df_join_dates = df_join_dates.astype("O")

        df = df_join_dates.merge(df, on=["year", "week", "year_week"], how="left")
        df.iloc[:, 3] = df.iloc[:, 3].fillna(
            df[ID_list[3]].drop_duplicates(keep="first").values[0]
        )
        df.iloc[:, 4] = df.iloc[:, 4].fillna(
            df[ID_list[4]].drop_duplicates(keep="first").values[0]
        )
        df.iloc[:, 5] = df.iloc[:, 5].fillna(
            df[ID_list[5]].drop_duplicates(keep="first").values[0]
        )
        df.iloc[:, 6] = df.iloc[:, 6].fillna(
            df[ID_list[6]].drop_duplicates(keep="first").values[0]
        )
        df.iloc[:, 7] = df.iloc[:, 7].fillna(
            df[ID_list[7]].drop_duplicates(keep="first").values[0]
        )
        df.iloc[:, 8] = df.iloc[:, 8].fillna(
            df[ID_list[8]].drop_duplicates(keep="first").values[0]
        )

        try:
            df.loc[:, "dsp_type_vin"].fillna(
                df["dsp_type_vin"].mode().values[0], inplace=True
            )
            df.loc[:, "dsp_type_attr"].fillna(
                df["dsp_type_attr"].mode().values[0], inplace=True
            )
            df.loc[:, "dsp_type_nh"].fillna(
                df["dsp_type_nh"].mode().values[0], inplace=True
            )
        except:
            df.loc[:, "dsp_type_vin"].fillna("DSP 2.0", inplace=True)
            df.loc[:, "dsp_type_attr"].fillna("DSP 2.0", inplace=True)
            df.loc[:, "dsp_type_nh"].fillna("DSP 2.0", inplace=True)

        imputer = KNNImputer(
            n_neighbors=4, weights="distance", keep_empty_features=True
        )
        df.loc[:, feature_list] = imputer.fit_transform(df[feature_list])

        if df.isna().sum().sum() > 0:
            raise Exception("Input Dataset Still has some missing values")
        else:
            logger.debug(
                f"Nearest Neighbour Imputations completed for region {df.region.unique()[0]}, no missing value found"
            )

        return df

    def do_year_lagged_imputations(self, df, feature_list):
        """
        This is a helper function used in multi-processing to make prior year based imputations at station_pair, stage level

        Params:
        ----------------------------------------------------------------
        df: pd.DataFrame()
                        input dataset required to impute missing values using KNN algorithm
        feature_list: list
                        list of features to be used for imputations

        Returns:
        -------------------------
        None

        """
        df_temp = df.pivot(
            columns=["year"],
            index=["station_pair", "station_code", "dsp", "region", "stage"],
            values=feature_list,
        )

        if df.shape[0] is None:
            return None

        if df.year.nunique() == 1:
            return df_temp.stack(level=1).reset_index()

        for year in df.year.unique().tolist()[1:]:
            for col in feature_list:
                try:
                    df_temp.loc[:, (f"is_{col}_missing", year)] = np.where(
                        df_temp.loc[:, (col, year)].isna(), 1, 0
                    )
                    df_temp.loc[:, (col, year)].fillna(
                        df_temp.loc[:, (col, year - 1)], inplace=True
                    )
                    df_temp.loc[:, (col, year - 1)].fillna(
                        df_temp.loc[:, (col, year)], inplace=True
                    )
                except:
                    # there are some edge cases observed where a DSP doesn't have continous yearly data, we basically impute those values from prior to missing year
                    df_temp.loc[:, (f"is_{col}_missing", year)] = np.where(
                        df_temp.loc[:, (col, year)].isna(), 1, 0
                    )
                    df_temp.loc[:, (col, year)].fillna(
                        df_temp.loc[:, (col, year - 2)], inplace=True
                    )
                    df_temp.loc[:, (col, year - 2)].fillna(
                        df_temp.loc[:, (col, year)], inplace=True
                    )
        df_temp = df_temp.stack(level=1).reset_index()
        return df_temp

    def do_missing_value_imputations(self, df, ID, feature_list):
        """
        This function is used to fill feature missing values for feature (fills 16% missing values across feature space)

        Params:
        ----------------------------------------------------------------
        df: pd.DataFrame()
                        input dataset required to impute missing values using KNN algorithm
        ID: list
                        list of ID related columns
        feature_list: list
                        list of features to be used for imputations

        Returns:
        -------------------------
        None

        """
        df.drop(columns="completed_routes_max", inplace=True)

        # requested routes are what DSP requested to do (DSP Perferences)
        df.loc[df["requested_routes"] == 0, "requested_routes"] = df["completed_routes"]
        df.loc[df["requested_routes_max"] == 0, "requested_routes_max"] = df[
            "completed_routes_target"
        ]

        df.rename(
            columns={"completed_routes_target": "completed_routes_max"}, inplace=True
        )

        for metric in [
            "grounded_vans",
            "grounded_days",
            "delivered",
            "dispatched",
            "incentives",
            "ot_incentives",
            "attendance_incentives",
            "signon_incentives",
            "t6_boc_count",
            "t6_boc_need_cure_count",
            "t6_boc_failed_count",
            "t6_boc_open_count",
            "t6_isight_points",
            "t6_crboc_count",
            "t6_cr_boc_need_cure_count",
            "t6_cr_boc_failed_count",
            "t6_cr_boc_open_count",
            "t6_cr_isight_points",
            "dsp_peak_years",
            "station_pair_peak_years",
        ]:
            # df.loc[:, f"is_{metric}_missing"] = np.where(df[metric].isna(), 1, 0)
            df[metric].fillna(0, inplace=True)

        df = df.reindex(
            columns=ID + ["dsp_type_vin", "dsp_type_attr", "dsp_type_nh"] + feature_list
        )

        group_list = self.find_similar_dsps(df)
        with Pool(processes=8) as pool:
            result = pool.map(
                partial(
                    self.do_similarity_based_imputations,
                    feature_list=feature_list,
                    performance_stage_config_dict=self.stage_config_dict,
                    ID_list=ID,
                ),
                group_list,
            )

        df = pd.concat(result)
        df = df.sort_values(by=["station_pair", "year", "week"]).reset_index(drop=True)

        logger.debug("Missing Value Imputation Completed")
        return df

    @timer
    def create_input_data(
        self,
        ID,
        feature_list,
        country_code,
        program_type,
        local_download_path,
        s3_upload_path,
        s3_upload_flag,
    ):
        """
        This function is used to create the input data by stiching together multiple data sources from redshift tables by
        running preprocessing steps to be used for feature engineering

        Params:
        -----------------------
        None

        Returns:
        -----------------------
        data: Dataframe
                        Input dataset post preprocessing read to be used by Feature Handler
        attrition_date: Dataframe
                        Input attrition dataset post preprocessing to be used by Feature Handler

        Example:
        -----------------------
        input_data, attrition_data = create_input_data()

        """
        dsp_data = self.do_preprocessing_dsp_data()
        vin_data = self.do_preprocessing_vin_data()
        attrition_data = self.do_preprocessing_attrition_data()
        nh_data = self.do_preprocessing_nh_data()
        incentive_data = self.do_preprocessing_incentive_data()

        # nh_data.rename(columns={"station": "station_code"}, inplace=True)

        vin_data.rename(columns={"dsp_type": "dsp_type_vin"}, inplace=True)
        attrition_data.rename(columns={"dsp_type": "dsp_type_attr"}, inplace=True)
        nh_data.rename(columns={"dsp_type": "dsp_type_nh"}, inplace=True)

        input_data = (
            dsp_data.merge(
                incentive_data[
                    [
                        "year",
                        "week",
                        "station_code",
                        "dsp_code",
                        "station_pair",
                        "country_code",
                        "total_incentives",
                        "ot_incentives",
                        "attendance_incentives",
                        "signon_incentives",
                    ]
                ],
                on=[
                    "year",
                    "week",
                    "station_code",
                    "dsp_code",
                    "station_pair",
                    "country_code",
                ],
                how="left",
            )
            .merge(
                vin_data,
                on=[
                    "year",
                    "week",
                    "station_code",
                    "dsp_code",
                    "station_pair",
                    "country_code",
                    "program_type",
                ],
                how="left",
            )
            .merge(
                nh_data,
                on=[
                    "year_week",
                    "dsp_code",
                    "station_code",
                    "station_pair",
                    "country_code",
                ],
                how="left",
            )
            .merge(
                attrition_data,
                on=[
                    "year",
                    "week",
                    "station_pair",
                    "dsp_code",
                    "station_code",
                    "country_code",
                ],
                how="left",
            )
        )
        input_data.rename(columns={"total_incentives": "incentives"}, inplace=True)
        input_data = input_data.sort_values(
            by=["station_pair", "year", "week"]
        ).reset_index(drop=True)
        input_data = self.merge_dsp_quality_score(input_data)
        input_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        input_data = self.do_missing_value_imputations(input_data, ID, feature_list)

        input_data.loc[:, "dsp_type"] = input_data.dsp_type_nh
        input_data.loc[:, "dsp_type"] = np.where(
            input_data.dsp_type.isna(), input_data.dsp_type_vin, input_data.dsp_type
        )
        input_data.loc[:, "dsp_type"] = np.where(
            input_data.dsp_type.isna(), input_data.dsp_type_attr, input_data.dsp_type
        )
        input_data.loc[:, "dsp_type"] = np.where(
            input_data.dsp_type.isin(
                ["DSP 2.0", "Migrated 2.0", "Migrating", "Wagon Wheel 2.0"]
            ),
            "DSP 2.0",
            np.where(
                input_data.dsp_type.isin(["Walker 2.0", "Walker"]),
                "Walker",
                np.where(input_data.dsp_type.isin(["DSP 1.0"]), "DSP 1.0", "DSP 2.0"),
            ),
        )

        input_data = input_data[input_data.country_code.isin(country_code)]
        input_data = input_data[input_data.program_type.isin(program_type)].reset_index(
            drop=True
        )
        input_data = input_data.round(7)

        if input_data.shape[0] < 200:
            raise Exception("Input Data has less than 200 rows")

        input_data.to_parquet(
            os.path.join(local_download_path, "dsp_peak_scaling_input_data.pqt"),
            index=False,
        )
        if s3_upload_flag:
            self.upload_to_s3(
                local_download_path=os.path.join(
                    local_download_path, "dsp_peak_scaling_input_data.pqt"
                ),
                s3_upload_path=s3_upload_path,
            )

        # take out attrition data for development purpose - not used in production
        # attrition_data.to_csv("input_files/attrition_data_processed.csv", index=False)

        logger.debug("Create input data is complete.")
        print()


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
    prediction_length = run_week_length + prediction_length
    run_year_week = run_year_week[0]
    run_week = int(run_week[0])

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

        stage1_config = df_year_week[
            df_year_week.time_period == stage1_peak_length
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
    run_date,
    mode,
    bucket,
    etl_prefix,
    country_code,
    program_type,
):
    """
    This function is to used run the model sequentially

    """

    s3_upload_flag = False
    if mode == "Prod":
        s3_upload_flag = True
    else:
        s3_upload_flag = False

    prediction_length = 6

    performance_stage_config_dict = generate_stages(
        prediction_length, run_date, stage_type="Performance"
    )

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
        }

    # Processed objects local/s3 upload paths
    input_file_path = os.path.join(os.getcwd(), f"input_files/{run_date}/")
    s3_dsp_data_upload_path = f"{etl_prefix}/dsp_data.pqt"
    s3_input_upload_path = f"input_files/publish_date={run_date}/input_data.pqt"

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
        "completed_routes_sum",
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
        "requested_routes_sum",
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
    ]

    # Data Loading and Preprocessing
    p_handler = PreprocessHandler(
        s3_object_path_dict=s3_object_path_dict,
        stage_config_dict=performance_stage_config_dict,
        run_date=run_date,
        bucket=bucket,
    )
    # p_handler.do_download_from_s3(s3_object_path_dict=s3_object_path_dict)

    p_handler.create_dsp_data(
        local_file_path=input_file_path,
        s3_upload_path=s3_dsp_data_upload_path,
        s3_upload_flag=s3_upload_flag,
    )

    p_handler.create_input_data(
        ID=ID_list,
        feature_list=feature_list,
        country_code=country_code,
        program_type=program_type,
        local_download_path=input_file_path,
        s3_upload_path=s3_input_upload_path,
        s3_upload_flag=s3_upload_flag,
    )


if __name__ == "__main__":
    run_id = sys.argv[1]  # unique run id assigned by the batch job
    run_date = sys.argv[2]
    mode = sys.argv[3]
    model_version = sys.argv[4]
    bucket = sys.argv[5]
    etl_prefix = sys.argv[6]
    region_code = sys.argv[7]
    program_type = sys.argv[8]

    if region_code == "NA":
        country_code = ["US", "CA"]
    else:
        raise Exception(
            f"Region Code {region_code} not a valid input, please provide values in ['NA', 'EU', 'FE', 'IN']"
        )

    run_date_format = "%Y-%m-%d"
    run_date = datetime.strptime(run_date, run_date_format).date()
    program_type = [program_type]

    main(
        run_date,
        mode,
        bucket,
        etl_prefix,
        country_code,
        program_type,
    )
