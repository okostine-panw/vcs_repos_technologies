# Import necessary libraries and modules
import configparser
import json
import os
import re
import time
from datetime import datetime
from pprint import pprint
import jmespath
import pandas as pd
import numpy as np

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests.exceptions import HTTPError

import psutil
import random
import traceback
from halo import Halo
from colorama import Fore

# Record memory usage before running the function
start_memory = psutil.virtual_memory().used
max_memory_used = start_memory

# Set Pandas display option to show full column width
pd.set_option('display.max_colwidth', None)
# Supress the SettingWithCopyWarning pandas warning
pd.set_option('mode.chained_assignment', None)

py_filename = os.path.basename(__file__)

# Make sure your instance value below which is part of the .ini filename is in the same directory
# Example API_config-{instance}.ini

instance = 'pso' # Replace this with your cutom instance name

# Define file names
# file_name_repos = f"Code-repos-technologies-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.csv"
file_name_repos = f"{py_filename}-{instance}-Code-repos-technologies_{datetime.now().strftime('%Y%m%d%H%M')}.csv"

# Disable urllib3 warnings to avoid warnings when using requests (useful when working with proxies)
requests.packages.urllib3.disable_warnings()
s = requests.Session()
s.timeout = 120
# Create a session with retry logic
# Include 400-level status codes (e.g., 400 Bad Request) in the retry mechanism
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[400, 500, 502, 503, 504])
s.mount('http://', HTTPAdapter(max_retries=retries))
s.mount('https://', HTTPAdapter(max_retries=retries))
# s.verify = 'test.crt' # Include your cert bundle here if using a proxy/vpn
start_time = time.time()

global random_number # debugging
random_number = random.randint(1000, 9999)


# Define a function to load API configuration from an INI file
def load_api_config(iniFilePath):
    if not os.path.exists(iniFilePath):
        # Exit if the configuration file does not exist
        return_error("Config file " + iniFilePath + " does not exist")
    iniFileParser = get_parser_from_sections_file(iniFilePath)
    api_config = {}
    api_config['BaseURL'] = read_value_from_sections_file_and_exit_if_not_found(iniFilePath, iniFileParser, 'URL',
                                                                                'URL')
    api_config['AccessKey'] = read_value_from_sections_file_and_exit_if_not_found(iniFilePath, iniFileParser,
                                                                                  'AUTHENTICATION', 'ACCESS_KEY_ID')
    api_config['SecretKey'] = read_value_from_sections_file_and_exit_if_not_found(iniFilePath, iniFileParser,
                                                                                  'AUTHENTICATION', 'SECRET_KEY')
    return api_config
# Define a function to perform initial login to the API
def login(api_config):
    global token
    action = "POST"
    url = api_config['BaseURL'] + "/login"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        'username': api_config['AccessKey'],
        'password': api_config['SecretKey'],
    }
    apiResponse = run_api_call_with_payload(action, url, headers, payload)
    authentication_response = apiResponse.json()
    token = authentication_response['token']
    return token


# Define a function to print an error message and exit the script
def return_error(message):
    print("\nERROR: " + message)
    exit(1)
# Define a function to parse configuration from sections within a file
def get_parser_from_sections_file(file_name):
    file_parser = configparser.ConfigParser()
    try:
        # Attempt to read and parse the file
        file_parser.read(file_name)
    except (ValueError, configparser.MissingSectionHeaderError, configparser.DuplicateOptionError,
            configparser.DuplicateOptionError):
        # Handle exceptions if the file format is improper
        return_error("Unable to read file " + file_name)
    return file_parser
# Define a function to read a value from sections within a file
def read_value_from_sections_file(file_parser, section, option):
    value = {}
    value['Exists'] = False
    if file_parser.has_option(section, option):
        # Check if the section and option exist in the file
        value['Value'] = file_parser.get(section, option)
        if not value['Value'] == '':
            # Check if the value is not blank (properly updated)
            value['Exists'] = True
    return value
# Define a function to read a value from sections within a file and exit if not found
def read_value_from_sections_file_and_exit_if_not_found(file_name, file_parser, section, option):
    value = read_value_from_sections_file(file_parser, section, option)
    if not value['Exists']:
        # Exit the script if the section and option are not found in the file
        return_error("Section \"" + section + "\" and option \"" + option + "\" not found in file " + file_name)
    return value['Value']
# Define a function to handle API response status
def handle_api_response(apiResponse):
    status = apiResponse.status_code
    if (status != 200):
        if (status == 401):
            token = login(api_config)
        else:
            # Handle API call failure with HTTP response status code
            return_error(Fore.YELLOW + "API call failed with HTTP response " + str(status))

# Define a function to run an API call with payload
def run_api_call_with_payload(action, url, headers_value, payload):
    apiResponse = s.request(action, url, headers=headers_value, data=json.dumps(payload), verify=False)
    # Perform the API call and handle the response
    handle_api_response(apiResponse)
    return apiResponse
# Define a function to run an API call without payload
def run_api_call_without_payload(action, url, headers_value):
    apiResponse = s.request(action, url, headers=headers_value, verify=False)
    # Perform the API call and handle the response
    handle_api_response(apiResponse)
    return apiResponse

# Define a function to retrieve scan_info results for native standards
def response_repos_info():
    url = f"{repos_info_url}"
    token = login(api_config)
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Accept': 'application/json; charset=UTF-8',
        'x-redlock-auth': token
    }
    payload = {
            "filters": {},
            "pageConfig": {
                "page": 1,
                "pageSize": 250
            }
        }
    # pprint(payload)
    response_json = []
    # response_json = requests.request("GET", url, headers=headers, data=payload).json()['resources']
    response = s.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        print(f"Success")
        try:
            response_json = response.json()
            # pprint(response_json)
            # Continue processing the response
        except json.decoder.JSONDecodeError:
            print(Fore.YELLOW + "JSONDecodeError: Response does not contain valid JSON data")
    else:
        print(Fore.RED + f" Native API request failed with status code {response.status_code}: {response.text}")
    return (pd.json_normalize([item for item in response_json]))

def df_to_xls(df):
    to_xls = pd.DataFrame(columns=['importanceScore', 'type', 'totalCommitsCount', 'contributorsCount', 'defaultBranch',
                                   'provider', 'name',
                                   'workspaceName', 'category', 'technology', 'detectedDate', 'percentage',
                                   'issues.SCA.TOTAL', 'issues.IAC.TOTAL', 'issues.SECRETS.TOTAL', 'issues.SAST.TOTAL',
                                   'repositorySize', 'url', 'ciFiles', 'lastUpdated', 'isArchived', 'Supported',
                                   'issues.SCA.CRITICAL', 'issues.SCA.HIGH', 'issues.SCA.MEDIUM', 'issues.SCA.LOW', 'issues.SCA.INFO',
                                   'issues.IAC.CRITICAL', 'issues.IAC.HIGH', 'issues.IAC.MEDIUM', 'issues.IAC.LOW', 'issues.IAC.INFO',
                                   'issues.SECRETS.CRITICAL', 'issues.SECRETS.HIGH', 'issues.SECRETS.MEDIUM', 'issues.SECRETS.LOW', 'issues.SECRETS.INFO',
                                   'issues.SAST.CRITICAL', 'issues.SAST.HIGH', 'issues.SAST.MEDIUM', 'issues.SAST.LOW', 'issues.SAST.INFO'])
    # from: https://github.com/bridgecrewio/checkov/blob/main/checkov/common/models/consts.py
    # SUPPORTED_FILE_EXTENSIONS = [".tf", ".yml", ".yaml", ".json", ".template", ".bicep", ".hcl"]
    # SUPPORTED_PACKAGE_FILES = {
    #     "bower.json",
    #     "build.gradle",
    #     "build.gradle.kts",
    #     "go.sum",
    #     "gradle.properties",
    #     "METADATA",
    #     "npm-shrinkwrap.json",
    #     "package.json",
    #     "package-lock.json",
    #     "pom.xml",
    #     "requirements.txt"
    # }
    # SUPPORTED_FILES = SUPPORTED_PACKAGE_FILES.union({"Dockerfile"})
    #
    # DEPENDENCY_TREE_SUPPORTED_FILES = {"yarn.lock", "Gemfile", "Gemfile.lock", "go.mod", "paket.dependencies", "paket.lock", "packages.config", "composer.json", "composer.lock"}
    #
    # SCANNABLE_PACKAGE_FILES_EXTENSIONS = {".csproj"}
    #
    # SCANNABLE_PACKAGE_FILES = SUPPORTED_PACKAGE_FILES | DEPENDENCY_TREE_SUPPORTED_FILES
    #
    # SAST_SUPPORTED_FILE_EXTENSIONS = {
    #     SastLanguages.JAVA: ['.java'],
    #     SastLanguages.JAVASCRIPT: ['.js'],
    #     SastLanguages.TYPESCRIPT: ['.ts'],
    #     SastLanguages.PYTHON: ['.py'],
    #     SastLanguages.GOLANG: ['.go']
    # }
    supported_Programming = [
        "Java", "JavaScript", "Python", "Dockerfile", "PowerShell", "Shell", "Makefile","Android","TypeScript","Batchfile"
    ]
    supported_PackageManager = [
        "Maven", "NPM", "PythonSetup", "Gradle", "Composer", "GOPackageManager", "dockerfile", "Yarn", "Bower", "RubyGems", "NuGet", "Dot-NetFramework"
    ]
    supported_Devops = [
        "HCL", "ArmTemplate", "Kubernetes", "CloudFormation", "Helm", "Ansible", "Kustomize", "DockerCompose"
    ]
    if any('categorizedTechnologies' in col for col in df.columns):
        data = []
        names_encountered = set()  # Track encountered names
        for index, row in df.iterrows():
            name = row['name']
            if name not in names_encountered:
                names_encountered.add(name)  # Add the name to the set
                for column in df.columns:
                    if 'categorizedTechnologies' in column:
                        parts = column.split('.')
                        category = parts[1]
                        technology = parts[2]
                        detected_date_column = f'categorizedTechnologies.{category}.{technology}.detectedDate'
                        detected_date = row.get(detected_date_column)
                        # print(f"{name}-{category}-{technology} Percentage: {detected_date}")
                        custom_repo_score = row.get('importanceScore.customRepoScore')
                        repo_score_value = row.get('importanceScore.repoScore')
                        if pd.notna(custom_repo_score):
                            repo_score = custom_repo_score
                        else:
                            repo_score = repo_score_value
                        # Extract contributor names from the list of contributors and join them into a comma-separated string
                        contributor_names = ','.join(contributor.get('name', '') for contributor in row.get('contributors', []))
                        if not pd.isna(detected_date):
                            percentage_column = f'categorizedTechnologies.{category}.{technology}.percentage'
                            percentage = round(row.get(percentage_column), 2)
                            # print(f"{detected_date} {name}-{category}-{technology} Percentage:{percentage_column} -  {percentage} - {contributor_names}")
                            supported = 'Yes' if (category != "Programming" and category != "PackageManager" and category != "Devops") else (
                                'Yes' if technology in supported_Programming or technology in supported_PackageManager or technology in supported_Devops else 'No')

                            data.append({
                                'type': row['privacyLevel'],
                                'provider': row['provider'],
                                'workspaceName': row['workspaceName'],
                                'name': row['name'],
                                'importanceScore': repo_score,
                                'category': category,
                                'technology': technology,
                                'Supported': supported,
                                'percentage': percentage,
                                'issues.IAC.TOTAL': '' if row.get('issues.IAC.TOTAL') == 0 else int(row.get('issues.IAC.TOTAL')) if pd.notna(row.get('issues.IAC.TOTAL')) else '',
                                'issues.SCA.TOTAL': '' if row.get('issues.IAC.TOTAL') == 0 else int(row.get('issues.SCA.TOTAL')) if pd.notna(row.get('issues.SCA.TOTAL')) else '',
                                'issues.SECRETS.TOTAL': '' if row.get('issues.IAC.TOTAL') == 0 else int(row.get('issues.SECRETS.TOTAL')) if pd.notna(row.get('issues.SECRETS.TOTAL')) else '',
                                'issues.SAST.TOTAL': '' if row.get('issues.IAC.TOTAL') == 0 else int(row.get('issues.SAST.TOTAL')) if pd.notna(row.get('issues.SAST.TOTAL')) else '',
                                'repositorySize': round(row['repositorySize'], 2),  # Round to 2 decimal digits
                                'url': row['url'],
                                'ciFiles': row['ciFiles'],
                                'isArchived': row['isArchived'],
                                'defaultBranch': row['defaultBranch'],
                                'totalCommitsCount': '' if pd.isna(row['totalCommitsCount']) or row['totalCommitsCount'] == 0 else int(row['totalCommitsCount']),
                                'contributorsCount': '' if pd.isna(row['contributorsCount']) or row['contributorsCount'] == 0 else int(row['contributorsCount']),
                                'contributors': contributor_names,
                                'detectedDate': detected_date,
                                'lastUpdated': row['lastUpdated'],
                                'issues.SCA.CRITICAL': '' if pd.isna(row.get('issues.SCA.CRITICAL', 0)) or row.get('issues.SCA.CRITICAL', 0) == 0 else int(row.get('issues.SCA.CRITICAL', 0)),
                                'issues.SCA.HIGH': '' if pd.isna(row.get('issues.SCA.HIGH', 0)) or row.get('issues.SCA.HIGH', 0) == 0 else int(row.get('issues.SCA.HIGH', 0)),
                                'issues.SCA.MEDIUM': '' if pd.isna(row.get('issues.SCA.MEDIUM', 0)) or row.get('issues.SCA.MEDIUM', 0) == 0 else int(row.get('issues.SCA.MEDIUM', 0)),
                                'issues.SCA.LOW': '' if pd.isna(row.get('issues.SCA.LOW', 0)) or row.get('issues.SCA.LOW', 0) == 0 else int(row.get('issues.SCA.LOW', 0)),
                                'issues.SCA.INFO': '' if pd.isna(row.get('issues.SCA.INFO', 0)) or row.get('issues.SCA.INFO', 0) == 0 else int(row.get('issues.SCA.INFO', 0)),
                                'issues.IAC.CRITICAL': '' if pd.isna(row.get('issues.IAC.CRITICAL', 0)) or row.get('issues.IAC.CRITICAL', 0) == 0 else int(row.get('issues.IAC.CRITICAL', 0)),
                                'issues.IAC.HIGH': '' if pd.isna(row.get('issues.IAC.HIGH', 0)) or row.get('issues.IAC.HIGH', 0) == 0 else int(row.get('issues.IAC.HIGH', 0)),
                                'issues.IAC.MEDIUM': '' if pd.isna(row.get('issues.IAC.MEDIUM', 0)) or row.get('issues.IAC.MEDIUM', 0) == 0 else int(row.get('issues.IAC.MEDIUM', 0)),
                                'issues.IAC.LOW': '' if pd.isna(row.get('issues.IAC.LOW', 0)) or row.get('issues.IAC.LOW', 0) == 0 else int(row.get('issues.IAC.LOW', 0)),
                                'issues.IAC.INFO': '' if pd.isna(row.get('issues.IAC.INFO', 0)) or row.get('issues.IAC.INFO', 0) == 0 else int(row.get('issues.IAC.INFO', 0)),
                                'issues.SECRETS.CRITICAL': '' if pd.isna(row.get('issues.SECRETS.CRITICAL', 0)) or row.get('issues.SECRETS.CRITICAL', 0) == 0 else int(row.get('issues.SECRETS.CRITICAL', 0)),
                                'issues.SECRETS.HIGH': '' if pd.isna(row.get('issues.SECRETS.HIGH', 0)) or row.get('issues.SECRETS.HIGH', 0) == 0 else int(row.get('issues.SECRETS.HIGH', 0)),
                                'issues.SECRETS.MEDIUM': '' if pd.isna(row.get('issues.SECRETS.MEDIUM', 0)) or row.get('issues.SECRETS.MEDIUM', 0) == 0 else int(row.get('issues.SECRETS.MEDIUM', 0)),
                                'issues.SECRETS.LOW': '' if pd.isna(row.get('issues.SECRETS.LOW', 0)) or row.get('issues.SECRETS.LOW', 0) == 0 else int(row.get('issues.SECRETS.LOW', 0)),
                                'issues.SECRETS.INFO': '' if pd.isna(row.get('issues.SECRETS.INFO', 0)) or row.get('issues.SECRETS.INFO', 0) == 0 else int(row.get('issues.SECRETS.INFO', 0)),
                                'issues.SAST.CRITICAL': '' if pd.isna(row.get('issues.SAST.CRITICAL', 0)) or row.get('issues.SAST.CRITICAL', 0) == 0 else int(row.get('issues.SAST.CRITICAL', 0)),
                                'issues.SAST.HIGH': '' if pd.isna(row.get('issues.SAST.HIGH', 0)) or row.get('issues.SAST.HIGH', 0) == 0 else int(row.get('issues.SAST.HIGH', 0)),
                                'issues.SAST.MEDIUM': '' if pd.isna(row.get('issues.SAST.MEDIUM', 0)) or row.get('issues.SAST.MEDIUM', 0) == 0 else int(row.get('issues.SAST.MEDIUM', 0)),
                                'issues.SAST.LOW': '' if pd.isna(row.get('issues.SAST.LOW', 0)) or row.get('issues.SAST.LOW', 0) == 0 else int(row.get('issues.SAST.LOW', 0)),
                                'issues.SAST.INFO': '' if pd.isna(row.get('issues.SAST.INFO', 0)) or row.get('issues.SAST.INFO', 0) == 0 else int(row.get('issues.SAST.INFO', 0)),
                            })
            else:
                data.append({
                    'type': row['privacyLevel'],
                    'totalCommitsCount': row['totalCommitsCount'],
                    'isArchived': row['isArchived'],
                    'defaultBranch': row['defaultBranch'],
                    'url': row['url'],
                    'ciFiles': row['ciFiles'],
                    'repositorySize': round(row['repositorySize'], 2),  # Round to 2 decimal digits
                    'lastUpdated': row['lastUpdated'],
                    'provider': row['provider'],
                    'name': row['name'],
                    'contributorsCount': row['contributorsCount'],
                    'contributors': contributor_names,
                    'workspaceName': row['workspaceName'],
                    'detectedDate': row.get('detectedDate'),
                    'percentage': row.get('percentage'),
                })

        to_xls = pd.DataFrame(data)
        # Remove rows where detectedDate is empty
        to_xls = to_xls[to_xls['detectedDate'].notna()]
        # Convert all list-like columns to strings
        to_xls = to_xls.apply(lambda x: x.map(lambda y: str(y) if isinstance(y, (list, dict)) else y))
        # Drop duplicates
        to_xls = to_xls.drop_duplicates()

    else:
        # If there are no categorized technologies, return the original dataframe
        to_xls = df[df['detectedDate'].notna()]
        # Drop duplicates
        to_xls = to_xls.drop_duplicates()

    return to_xls

# to_xls = df_to_xls(df)

def get_repos():
    global random_number
    # Define a function to append non-empty dataframes to the output
    df = response_repos_info()
    # df.to_csv(f"debug-df-{random_number}.csv", index=False)
    result_repos = df_to_xls(df)
    # Debugging, please ignore
    # result_repos.to_csv(f"debug-result_repos{random_number}.csv", index=False)
    return result_repos

# Main script run
# Load API configuration from the INI file
api_config = load_api_config(f"API_config-{instance}.ini")
# Perform the first API call for authentication and store the token
token = login(api_config)
api_config['Token'] = token
# Define URLs used for API calls
base = api_config['BaseURL']
repos_info_url = api_config['BaseURL'] + "/bridgecrew/api/v1/vcs-repository/repositories"

# Define generic headers to be used for API calls
headers = {
    'Content-Type': 'application/json; charset=UTF-8',
    'Accept': 'application/json; charset=UTF-8',
    'x-redlock-auth': api_config['Token']
}
result_repos = get_repos()
result_repos.to_csv(file_name_repos, index=False)
print(f"Total Number of results: {len(result_repos)}")
# pprint (result_repos)
# results_repos.to_csv(file_name_repos, index=False)
# print(f"Total results rows: {len(results_repos)}")

# Record memory usage after running the function
end_memory = psutil.virtual_memory().used

# Calculate the maximum memory used
max_memory_used = max(max_memory_used, end_memory)
# Print the maximum memory used
print(f"Maximum memory used: {max_memory_used / (1024 ** 2):.2f} MB")
# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time_seconds = end_time - start_time
# Convert elapsed time to minutes
elapsed_time_minutes = elapsed_time_seconds / 60
# Print the script elapsed run time in minutes
print(Fore.YELLOW + f"Time taken: {elapsed_time_minutes:.2f} minutes")
