import os
import sys
import logging
import pandas as pd
from drain3 import TemplateMiner

logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_logs_with_drain3(log_file_path):
    template_miner = TemplateMiner()
    log_lines = []
    event_ids = []
    templates = {}
    with open(log_file_path, 'r') as f:
        for line in f:
            log_line = line.strip()
            result = template_miner.add_log_message(log_line)
            if result is not None:
                eid = result["cluster_id"]
                event_ids.append(eid)
                templates[eid] = result["template_mined"]
                log_lines.append(log_line)
    df = pd.DataFrame({"log": log_lines, "EventId": event_ids})
    return df, templates

def deeplog_df_transfer(df, event_id_map):
    # You may need to adjust this if your log files don't have Date/Time columns
    df['datetime'] = pd.to_datetime(df.index, unit='s')
    df = df[['datetime', 'EventId']]
    df['EventId'] = df['EventId'].apply(lambda e: event_id_map[e] if event_id_map.get(e) else -1)
    deeplog_df = df.set_index('datetime').resample('1min').apply(_custom_resampler).reset_index()
    return deeplog_df

def _custom_resampler(array_like):
    return list(array_like)

def deeplog_file_generator(filename, df):
    with open(filename, 'w') as f:
        for event_id_list in df['EventId']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n')

if __name__ == '__main__':
    ##########
    # Parser #
    ##########
    input_dir = 'C:/Users/farid/PycharmProjects/FaultDetectionProject/OpenStack/'
    output_dir = './openstack_result/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse each log file and save structured CSVs
    log_files = {
        "openstack_normal1.log": "openstack_normal1.log_structured.csv",
        "openstack_normal2.log": "openstack_normal2.log_structured.csv",
        "openstack_abnormal.log": "openstack_abnormal.log_structured.csv"
    }
    for log_name, csv_name in log_files.items():
        log_path = os.path.join(input_dir, log_name)
        df, templates = parse_logs_with_drain3(log_path)
        df.to_csv(os.path.join(output_dir, csv_name), index=False)

    ##################
    # Transformation #
    ##################
    df = pd.read_csv(f'{output_dir}/openstack_normal1.log_structured.csv')
    df_normal = pd.read_csv(f'{output_dir}/openstack_normal2.log_structured.csv')
    df_abnormal = pd.read_csv(f'{output_dir}/openstack_abnormal.log_structured.csv')

    # Build event_id_map from all unique EventIds
    all_event_ids = pd.concat([df['EventId'], df_normal['EventId'], df_abnormal['EventId']]).unique()
    event_id_map = {eid: i+1 for i, eid in enumerate(all_event_ids)}

    logger.info(f'length of event_id_map: {len(event_id_map)}')

    #########
    # Train #
    #########
    deeplog_train = deeplog_df_transfer(df, event_id_map)
    deeplog_file_generator('train', deeplog_train)

    ###############
    # Test Normal #
    ###############
    deeplog_test_normal = deeplog_df_transfer(df_normal, event_id_map)
    deeplog_file_generator('test_normal', deeplog_test_normal)

    #################
    # Test Abnormal #
    #################
    deeplog_test_abnormal = deeplog_df_transfer(df_abnormal, event_id_map)
    deeplog_file_generator('test_abnormal', deeplog_test_abnormal)
    