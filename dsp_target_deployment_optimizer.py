import sys

sys.path.insert(0, "/home/ec2-user/SageMaker/dsp_capacity_forecast/")
import dsp_capacity_forecast as dcpm
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
from loguru import logger
import boto3
import json
import psycopg2
import uuid4
import re
import gurobipy as gp
from gurobipy import GRB


class DataLoader(object):
    def __init__(self, bucket):
        self.bucket = bucket

    def get_conn(self, cluster_name, secret_name):
        """
        The function is used to run queries on AMZL Analytics Redshift Cluster using RJDBC driver

        Params:
        -----------------------
        cluster_name = redshift cluster name (choose from amzlanalytics, amzl-bia-compute)

        Returns:
        -----------------------
        connection = connection object to run queries on the Redshift Cluster

        """

        # secret_name = secret_name
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

class DataWrangler(object):
    def __init__(self, s3_path_dict, bucket, execution_week, model_run_year_week, run_date):
        self.path_dict = s3_path_dict
        self.station = pd.DataFrame()
        self.station_dsp = pd.DataFrame()
        self.execution_week = execution_week
        self.model_run_year_week = model_run_year_week
        self.model_run_date = run_date

        self.dh = DataLoader(bucket)

    def create_dsp_data(self):
        """
        The function is used to create the dsp data by stiching together multiple data sources from redshift tables to create DSP data

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
            "scorecard_path",
        ]:
            temp = self.dh.read_csv_from_s3(self.path_dict[metric])
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

        if data.shape[0] < 200:
            raise Exception("DSP Data has less than 200 rows")

        logger.debug(f"Action - Creating DSP Data is complete. DSP Data Shape is {data.shape}")
        print()
        return data

    def do_preprocess_dsp_data(self, model_run_year_week):
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
        df = self.create_dsp_data()
        df = df.loc[df.year >= 2021]
        # df = df.drop(columns=["dsp_type"])
        df = df.astype(
            {
                "stage_end": "datetime64[ns]",
                "stage_start": "datetime64[ns]",
            }
        )
        df["week"] = ["%02d" % x for x in df.week]
        df.loc[:, "year_week"] = df["year"].astype(str) + "-" + df["week"].astype(str)

        # df = df.drop(
        #     columns=[
        #         "dvcr_metric",
        #         "safe_driving_metric",
        #     ]
        # )

        numeric_columns = df.select_dtypes(include="number").columns.tolist()[3:]
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
        df = df[df.year_week == model_run_year_week].reset_index(drop=True)
        df.rename(columns={"station_code":"station"}, inplace=True)

        # station_region = self.dh.read_csv_from_s3(
        #     self.s3_object_path_dict["station_region_path"]
        # )
        # df = df.merge(
        #     station_region,
        #     left_on=["station_code", "country_code"],
        #     right_on=["location_id", "country_code"],
        #     how="left",
        # ).iloc[:, :-1]

        if df.shape[0] < 200:
            raise Exception("DSP Data not sufficient, has less than 200 rows")
        else:
            shape = df.shape
            logger.debug(f"Preprocssing DSP Data is complete. DSP Data shape is {shape}")

        return df

    def do_preprocess_lrp_plan_data(self, execution_week):
        """
        This function is used to preprocess the lrp_plan data for creating the station universe for optimization model
        
        """
        df = self.dh.read_csv_from_s3(self.path_dict["lrp_plan_path"])
        df["year_week"] = df.execution_year.astype(str) + "-" + df.execution_week_num.astype(str)
        df = df[df["year_week"] == execution_week].reset_index(drop=True)
        df = df.groupby(["plan_publish_date", 
                         "country", 
                         "station", 
                         "year_week", 
                         "execution_year", 
                         "execution_week_num"]).agg({"lrp_volume":"sum", 
                                                     "lrp_spr":"sum"}).reset_index()
        df["lrp_route_target"] = np.round(df["lrp_volume"] / df["lrp_spr"].fillna(df.lrp_spr.mean()), 2)

        return df

    def do_preprocess_okami_data(self, execution_week):
        """
        This function is used to preprocess okami offered route count data
        """

        df = self.dh.read_csv_from_s3(self.path_dict["okami_path"])
        df = df[df.reporting_year_week == execution_week].reset_index(drop=True)
        df.rename(columns={"route_count":"okami_offered_routes"}, inplace=True)
        return df

    def do_preprocess_capacity_data(self, execution_week):
        """
        This function is used to preprocess the capacity data from DCPM model
        """
        dh_temp = DataLoader("dsp-capacity-forecast")
        date_list = pd.date_range(start=self.model_run_date-timedelta(91), periods=27, freq="W-WED")
        keys = []
        for date in date_list:
            try:
                k = dh_temp.read_s3_keys_from_prefix(f"inferences/{date.date()}/{execution_week}")
                if keys:
                    keys.extend(k)
                else:
                    keys = k
            except:
                continue

        result = []
        for k in keys:
            df = dh_temp.read_csv_from_s3(k)
            result.append(df)

        df = pd.concat(result)

        df[df.predictions_with_incentives == 0] = np.nan
        df = df[df.station_pair.notna()]
        df = df[df.publish_date == df.publish_date.min()] ## change this to max when we are running everything with respect to model run week
        df["station"] = df.station_pair.apply(lambda x: x.split('-')[0])
        # df = df.rename(columns={"dsp":"dsp_dcpm"})
        df = df[~df.dsp_type.isin(["Transfer Out", "Exiting"])].reset_index(drop=True)
        shape = df.shape
        logger.debug(f"Preprocessing DCPM capacity data is complete. Shape of the dataframe is {shape}")
        # df.predictions_with_incentives.fillna - at station pair level fill with moving average, at station level fill with moving average)
        return df

    def do_preprocess_geo_data(self):
        df = self.dh.read_csv_from_s3(self.path_dict["ds_lat_long_path"])
        df = df[df.country_code.isin(["US", "CA"])]
        df = df[df.latitude.notna()].reset_index(drop=True)
        df.rename(columns={"station_code":"station", "country_code":"country"}, inplace=True)
        stations = df.station.nunique()
        logger.debug(f"Preprocessing station geo data is completed. Total Number of stations are ({stations})")
        return df

    def create_input_data(self):
        lrp_plan_data = p.do_preprocess_lrp_plan_data(p.execution_week)
        okami_data = p.do_preprocess_okami_data(p.execution_week)
        dcpm_cf_data = p.do_preprocess_capacity_data(p.execution_week)
        dcpm_cf_data = dcpm_cf_data[["station", "dsp", "predictions_with_incentives"]]
        dsp_data = p.do_preprocess_dsp_data(p.model_run_year_week)
        dsp_data = dsp_data[["station_pair", "dsp_final_metric", "reliability", "station_pair_tenure"]]
        geo_data = p.do_preprocess_geo_data()

        station = lrp_plan_data.merge(geo_data, on=["station", "country"], how="left")
        station.loc[:, "launch_date"] = pd.to_datetime(station["launch_date"])
        station = station.drop_duplicates(keep='first').reset_index(drop=True)

        station_dsp = station.merge(okami_data, on=["station", "country"], how="left")
        df_temp = station[["station", "country", "year_week"]].merge(dcpm_cf_data, on=["station"], how="left")
        station_dsp = station_dsp.merge(df_temp, on=["station", "dsp", "country", "year_week"], how="left")
        station_dsp = station_dsp.merge(dsp_data, on=["station_pair"], how="left")
        station_dsp = station_dsp[station_dsp.okami_offered_routes > 0].reset_index(drop=True)
        station_dsp["predictions_with_incentives"] = station_dsp["predictions_with_incentives"].fillna(station_dsp["okami_offered_routes"])
        station_dsp = station_dsp.drop_duplicates(keep='first').reset_index(drop=True)
        station_dsp = station_dsp.sort_values(by=["station", "dsp"]).reset_index(drop=True)

        # station_dsp = station_dsp.merge(dsp_data, on=["station_pair",
        #                                               "country",
        #                                               "station",
        #                                               "dsp"], how="left")

        self.station = station.copy()
        self.station_dsp = station_dsp.copy()
        shape = station_dsp.shape
        logger.debug(f"Create input data completed. Shape of data = {shape}")


class DecisionMaker(object):
    def __init__(self, path_dict, bucket, execution_week, model_run_year_week, run_date):
        self.path_dict = path_dict
        self.p_handler = DataWrangler(path_dict,
                                         bucket,
                                         execution_week,
                                         model_run_year_week,
                                         run_date)
        self.p_handler.create_input_data()
        self.station = self.p_handler.station.copy()
        self.station_dsp = self.p_handler.station_dsp.copy()

    def find_undersolved_stations(self):
        station_dsp = self.station_dsp.copy()
        station_cap = station_dsp.groupby(["station"])[["okami_offered_routes", 
                                                "predictions_with_incentives"]].sum().astype("int32").reset_index()
        station_cap.rename(columns={"predictions_with_incentives": "station_capacity",
                                   "okami_offered_routes": "station_okami_offered"}, inplace=True)


        station_dsp = station_dsp.merge(station_cap, on="station", how="left")
        station_dsp["gap_station"] = station_dsp["station_capacity"] - station_dsp["lrp_route_target"]
        station_dsp["gap_okami"] = station_dsp["station_okami_offered"] - station_dsp["lrp_route_target"]
        station_dsp["is_under_solved"] = np.where(station_dsp.gap_station < 0 , 1, 0)

        if station_dsp.shape[0] < 600:
            raise Exception(f"Not enough rows in the dataframe to proceed")
        else:
            self.station_dsp = station_dsp.copy()
            logger.debug(f"Created a list of under-solved stations")

    def create_eligible_targets(self):
        station_dsp = self.station_dsp.copy()

        # DSP capacity gap check
        station_dsp["is_eligible_target"] = np.where(station_dsp.okami_offered_routes < station_dsp.predictions_with_incentives,
                                              1, 0)
        # DSP eligibility check (DSP 2.0 program)
        # station_dsp["is_eligible_target"] = np.where((station_dsp.dsp_latest_status_flag >= 2.0) & station_dsp.is_eligible_target > 0,
        #                                          1, station_dsp.is_eligible_target)
        # DSP Health check
        station_dsp["is_eligible_target"] = np.where(station_dsp.dsp_final_metric < station_dsp.dsp_final_metric.quantile(0.5), 
                                                  0, station_dsp.is_eligible_target)
        # DSP relibility check
        station_dsp["is_eligible_target"] = np.where(station_dsp.reliability < station_dsp.reliability.quantile(0.5),
                                                 0, station_dsp.is_eligible_target)

        # DSP present target state check
        dsp = station_dsp.groupby(['dsp']).station.nunique().reset_index()
        dsp = dsp[dsp.station > 1]
        station_dsp["is_eligible_target"] = np.where(station_dsp.dsp.isin(dsp.dsp), 0, station_dsp.is_eligible_target)

        # Station condition check
        station_dsp["is_eligible_target"] = np.where(station_dsp.is_under_solved > 0, 0, station_dsp.is_eligible_target)

        if station_dsp.shape[0] < 600:
            raise Exception(f"Not enough rows in the dataframe to proceed")
        else:
            self.station_dsp = station_dsp.copy()
            logger.debug(f"Create a list of eligible DSPs that can serve as targets")


class CostCalculator(object):
    def __init__(self, path_dict, bucket,
                                         execution_week,
                                         model_run_year_week,
                                         run_date):
        self.dm = DecisionMaker(path_dict, bucket,
                                         execution_week,
                                         model_run_year_week,
                                         run_date)
        self.dm.find_undersolved_stations()
        self.dm.create_eligible_targets()
        self.eligible_target_list = self.dm.station_dsp[self.dm.station_dsp.is_eligible_target==1]["dsp"].unique().tolist()
        self.unsolved_station_list = self.dm.station_dsp[self.dm.station_dsp.is_under_solved==1].station.unique().tolist()
        self.station = self.dm.station.copy()
        self.station_dsp = self.dm.station_dsp.copy()
        # self.undersolved_stations = list()
        # self.eligible_targets = list()
        self.roles = ["popup", "pinnacle", "transfer", "newdsps"]
        self.capacity_ask = None

    def print_model_stats(self):
        """
        This function is used to print some summary statistics for DetOr Model
        """

        df_temp = self.station_dsp[self.station_dsp.is_under_solved == 1]
        df_temp = df_temp.groupby("station").agg({"gap_station":"mean"}).reset_index()
        self.capacity_ask = dict(zip(df_temp.station, np.round(np.abs(df_temp.gap_station))))

        model_stats = {
                    "stations": self.station_dsp.station.nunique(),
                    "dsps": self.station_dsp.dsp.nunique(),
                    "unsolved_stations": len(self.unsolved_station_list),
                    "targets": len(self.eligible_target_list),
                    "capacity_ask": sum(self.capacity_ask.values())
                }

        print(f"Total Number of Stations: {model_stats['stations']}")
        print(f"Total Number of DSPs: {model_stats['dsps']}")
        print(f"Total Number Eligible Targets: {model_stats['targets']}")
        print(f"Total Number of unsolved stations: {model_stats['unsolved_stations']}")
        print()

    def define_solver_components(self):
        """
        This function is used to define solver objects to be utilized for creating solver objects
        """
        # self.undersolved_stations = list(
        #     self.station_dsp[self.station_dsp.is_under_solved == 1].station.unique()
        # )
        # self.eligible_targets = list(self.station_dsp[self.station_dsp.is_eligible_target == 1].dsp.unique())

        dsp_role_stations = []
        for d in self.eligible_target_list:
            for r in self.roles:
                for s in self.unsolved_station_list:
                    if dsp_role_stations:
                        dsp_role_stations.append((d,r,s))
                    else:
                        dsp_role_stations = [(d,r,s)]

        print(f"Total decision combinations: {len(dsp_role_stations)}")
        return dsp_role_stations

    def calculate_dcpm_capacities(self):
        pass

    def define_solver_objects(self):
        """
        This function is used to define lookup tuples to be utilized for creating solver objects
        """
        dsp_role_stations = self.define_solver_components()

        # Lookup tables
        dsp_role_station_combo, deployment_fixed_cost = gp.multidict(
            dict(
                zip(
                    dsp_role_stations, 
                    np.random.normal(loc=10000.0, scale=500.0, size=len(dsp_role_stations)).clip(5000.0, 15000.0),
                )
            )
        )

        dsp_role_station_combo, capacity_projection = gp.multidict(
            dict(
                zip(
                    dsp_role_stations,
                    np.random.normal(
                        loc=20.0, scale=2.5, size=len(dsp_role_stations)
                    ).clip(15.0, 25.0),
                )
            )
        )
        return dsp_role_station_combo, deployment_fixed_cost, capacity_projection


class OptimizerPro(object):
    def __init__(self, path_dict, bucket,
                                         execution_week,
                                         model_run_year_week,
                                         run_date):
        self.cc = CostCalculator(path_dict, bucket,
                                         execution_week,
                                         model_run_year_week,
                                         run_date)
        dsp_role_station_combo, deployment_fixed_cost, capacity_projection = self.cc.define_solver_objects()
        self.cc.print_model_stats()
        self.dsp_role_station_combo = dsp_role_station_combo
        self.deployment_fixed_cost = deployment_fixed_cost
        self.capacity_projection = capacity_projection
        self.eligible_target_list = self.cc.eligible_target_list
        self.model = None
        self.x = None

    def do_optimization(self):
        """
        This function is used to initialize the solve the optimization using model objects
        """
        # Initializing model
        m = gp.Model("Detor")
        m.setParam("LogFile", "model_output/detor_model.log")

        # Defining the decision variables
        self.x = m.addVars(self.dsp_role_station_combo, vtype=GRB.BINARY, name="capacity_assignment")

        # Defining Constraints
        # A DSP target combinations can be assigned to only one station
        station_assignment_constraint = m.addConstrs(
            (self.x.sum(D, '*', '*') <= 1 for D in self.eligible_target_list), name="target_station"
        )

        # capacity_ask_contraint = m.addConstr(
        #     (self.x.prod(self.capacity_projection) <= sum(self.cc.capacity_ask.values())), name="capacity_ask"
        # )

        station_capacity_ask_constraint = m.addConstrs(
            (
                sum(
                    [
                        self.x.select("*", "*", S)[i] * self.capacity_projection.select("*", "*", S)[i]
                        for i in range(len(self.capacity_projection.select("*", "*", S)))
                    ]
                )
                <= self.cc.capacity_ask[S]
                for S in self.cc.unsolved_station_list
            ), name="station_capacity_ask"
        )

        # Defining the objective function
        # Lets assume network 1 costs $1.5 and network 2 costs $3.5 to ship packages
        m.setObjective(3.5*200*7 * sum(self.cc.capacity_ask.values()) - 1.5*200*7 * self.x.prod(self.capacity_projection) + self.x.prod(self.deployment_fixed_cost), GRB.MINIMIZE)

        # Run Optimization
        m.optimize()

        m.write("model_output/detor_demo.lp")

        self.model = m

        logger.debug("Optimization Model Run completed Successfully")

    def print_model_artifacts(self):
        """
        This function is used to print and store model outputs
        """
        for v in self.model.getVars():
            if v.x > 1e-6:
                print(v.varName, v.x)
        selected_variables = [v.varName for v in self.model.getVars() if v.x > 1e-6]
        print(f"Total Decision Variables selected by solver: {len(selected_variables)}")
        capacity_added = self.x.prod(self.capacity_projection).getValue()
        fixed_cost_incurred = self.x.prod(self.deployment_fixed_cost).getValue()
        print(f"Total Capacity Added to the network = {capacity_added}")
        print(f"Total Fixed Cost Incurred = {fixed_cost_incurred}")
        

if __name__ == "__main__":
    # run_date = datetime.today().date()
    run_date = datetime(2023, 7, 12).date()
    run_week = run_date.isocalendar()[1]
    run_year = run_date.year
    model_run_year_week = str(run_year) + "-" + str(run_week)
    print({"Model run date": run_date,
           "Model run week": run_week,
           "Model run year": run_year,
           "Model run year week": model_run_year_week})

    mode = "Dev"
    execution_week = "2023-50"

    if mode == "Dev":
        s3_upload_flag = False
    else:
        s3_upload_flag = True

    bucket = "dsp-target-deployment-optimizer"

    # s3 download object paths (s3 versions stay in merge update)
    s3_object_path_dict = {
        "agd_path": "etl_files/detor_agd.csv000",
        "completed_routes_path": "etl_files/detor_completed_routes.csv000",
        "grounded_path": "etl_files/detor_grounded.csv000",
        "scorecard_path": "etl_files/detor_scorecard.csv000",
        "ds_lat_long_path": "etl_files/detor_ds_lat_long.csv000",
        "okami_path": "etl_files/detor_okami_offered_routes.csv000",
        "lrp_plan_path": "etl_files/detor_station_lrp_plan.csv000"
    }

    dcpm_local_path = os.path.join(os.getcwd(), "input_files/detor_dcpm_cf_data_stiched.csv")

    local_object_path_dict = {
        "volume_path": os.path.join(os.getcwd(), "input_files/scenario_planning_volume_forecast.csv"),
        "spr_path": os.path.join(os.getcwd(), "input_files/scenario_planning_spr_forecast.csv"),
        "okami_path": os.path.join(os.getcwd(), "input_files/scenario_planning_okami_plans.csv"),
        "geo_path": os.path.join(os.getcwd(), "input_files/scenario_planning_station_lat_long.csv"),
        "capacity_path": os.path.join(os.getcwd(), "input_files/dsp_peak_scaling_linear_predictions_48.csv"),
        "dsp_data_path": os.path.join(os.getcwd(), "input_files/dsp_data.csv")
        }

    op = OptimizerPro(s3_object_path_dict, bucket,
                                         execution_week,
                                         model_run_year_week,
                                         run_date)
    op.do_optimization()
    op.print_model_artifacts()
