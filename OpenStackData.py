import re
import os
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch

# Initialize Drain3 for log template extraction
persistence = FilePersistence("drain3_state.bin")
template_miner = TemplateMiner(persistence)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


# Function to parse log lines using Drain3
def parse_log_with_templates(file_path):
    parsed_logs = []
    with open(file_path, 'r') as file:
        for line in file:
            result = template_miner.add_log_message(line.strip())
            print(f"Drain3 Result: {result}")  # Debugging output
            parsed_logs.append({
                "log_line": line.strip(),
                "template": result["template_mined"],
                "parameters": result["parameter_list",[]]
            })
    return parsed_logs


if __name__ == "__main__":
    # Replace with your log file path
    log_file_path = "C:/Users/farid/PycharmProjects/FaultDetectionProject/OpenStack/openstack_normal1.log"
    
    # Check if the file exists
    if not os.path.exists(log_file_path):
        print(f"Error: The file '{log_file_path}' does not exist.")
    else:
        # Parse logs using Drain3
        logs = parse_log_with_templates(log_file_path)
        for log in logs:
            print(f"Log Line: {log['log_line']}")
            print(f"Template: {log['template']}")
            print(f"Parameters: {log['parameters']}")
            print("-" * 50)



""" 
# Prepare data: each log line as a text sample, with label 0 or 1
class LogDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

 """


""" def parse_log_line(line):
    pattern = r'^(nova-api\.log\.\d+\.\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) (\d+) (\w+) ([\w\.]+) \[req-([a-f0-9\-]+) ([a-f0-9]+) ([a-f0-9]+) - - -\] ([\d\.]+) "(GET|POST|PUT|DELETE) ([^"]+) HTTP/1\.1" status: (\d+) len: (\d+) time: ([\d\.]+)'
    match = re.match(pattern, line)
    if match:
        if match:
            log_file = match.group(1)
            log_timestamp = match.group(2)
            process_id = match.group(3)
            log_level = match.group(4)
            logger_name = match.group(5)
            request_id = match.group(6)
            tenant_id = match.group(7)
            user_id = match.group(8)
            client_ip = match.group(9)
            http_method = match.group(10)
            url = match.group(11)
            status_code = match.group(12)
            response_length = match.group(13)
            response_time = match.group(14)


        return {
            "log_file": log_file,
            "log_timestamp": log_timestamp,
            "process_id": process_id,
            "log_level": log_level,
            "logger_name": logger_name,
            "request_id": request_id,
            "tenant_id":tenant_id,
            "user_id": user_id,
            "client_ip": client_ip,
            "http_method": http_method,
            "url": url,
            "status_code": status_code,
            "response_length": response_length,
            "response_time": response_time
       
        }
 """
""" 
# Function to read and parse a log file
def read_log_file(file_path):
    
    parsed_logs = []
    with open(file_path, 'r') as file:
        for line in file:
            # print("Parsed line", line)
            parsed_line = parse_log_line(line.strip())
            if parsed_line:
                parsed_logs.append(parsed_line)
    return parsed_logs

if __name__ == "__main__":
    log_file_path = "C:/Users/farid/PycharmProjects/FaultDetectionProject/OpenStack/openstack_normal1.log"  # Replace with your log file path
    
    # Check if the file exists
    if not os.path.exists(log_file_path):
        print(f"Error: The file '{log_file_path}' does not exist.")
    else:
        logs = read_log_file(log_file_path)
        for log in logs:
          #  print("Parsed log:", log)
        #else:
         #   print("No logs were parsed.")
      """