from diagrams import Cluster, Diagram
from diagrams.programming.framework import React, Flask
from diagrams.onprem.client import User
from diagrams.azure.compute import VM
from diagrams.custom import Custom

with Diagram("Eye cancer recognition pipeline") as diag:
    with Cluster("Azure"):
        with Cluster("Webserver"):
            react = React("React")
            flask = Flask("Flask")
            pytorch = Custom("Pytorch", "./pytorch_logo.png")
            server = react >> flask >> pytorch
        azure = [VM("Azure VM"), server]

    User("User") >> react

diag
