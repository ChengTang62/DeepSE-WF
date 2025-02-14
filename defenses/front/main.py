import argparse
import configparser
import datetime
import logging
import os
import sys
import time
from os import makedirs
from os.path import join
from time import strftime

import constants as ct
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("Front")

def init_directories(section):
    if not os.path.exists(ct.RESULTS_DIR):
        makedirs(ct.RESULTS_DIR)
    timestamp = strftime("%m%d_%H%M")
    section_str = str(section).capitalize()
    output_dir = join(ct.RESULTS_DIR, f"Front_{section_str}_{timestamp}")
    makedirs(output_dir)
    return output_dir

def config_logger(args):
    log_file = sys.stdout if args.log == "stdout" else open(args.log, "w")
    ch = logging.StreamHandler(log_file)
    ch.setFormatter(logging.Formatter(ct.LOG_FORMAT))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

def parse_arguments():
    conf_parser = configparser.RawConfigParser()
    conf_parser.read(ct.CONFIG_FILE)
    parser = argparse.ArgumentParser(description="It simulates adaptive padding on a set of web traffic traces.")
    parser.add_argument("p", metavar="<traces path>", help="Path to the directory with the traffic traces to be simulated.")
    parser.add_argument("-format", metavar="<suffix of a file>", default="", help="suffix of a file.")
    parser.add_argument("-c", "--config", dest="section", metavar="<config name>", choices=conf_parser.sections(), default="default", help="Adaptive padding configuration.")
    parser.add_argument("--log", type=str, dest="log", metavar="<log path>", default="stdout", help="path to the log file. It will print to stdout by default.")
    args = parser.parse_args()
    config = dict(conf_parser._sections[args.section])
    config_logger(args)
    return args, config

def load_trace(fdir):
    df = pd.read_csv(
        fdir, 
        sep="\s+",  # Change from "\t" to "\s+" to handle inconsistent spaces
        header=None, 
        names=["time", "direction", "size"], 
        dtype={"time": float, "direction": int, "size": int}
    )
    return df.to_numpy()
def dump(trace, fname):
    global output_dir
    outfile = join(output_dir, fname)
    with open(outfile, "w") as fo:
        for packet in trace:
            line = "{:.4f}\t{}\t{}".format(packet[0], int(packet[1]), int(packet[2])) + ct.NL
            fo.write(line)

def simulate(fdir):
    if not os.path.exists(fdir):
        raise FileNotFoundError(fdir)
    np.random.seed(datetime.datetime.now().microsecond)
    trace = load_trace(fdir)
    trace = RP(trace)
    fname = fdir.split("/")[-1]
    dump(trace, fname)

def RP(trace):
    global client_dummy_pkt_num, server_dummy_pkt_num, min_wnd, max_wnd, start_padding_time
    global client_min_dummy_pkt_num, server_min_dummy_pkt_num
    
    client_wnd = np.random.uniform(min_wnd, max_wnd)
    server_wnd = np.random.uniform(min_wnd, max_wnd)
    
    client_dummy_pkt = np.random.randint(client_min_dummy_pkt_num, client_dummy_pkt_num) if client_min_dummy_pkt_num != client_dummy_pkt_num else client_dummy_pkt_num
    server_dummy_pkt = np.random.randint(server_min_dummy_pkt_num, server_dummy_pkt_num) if server_min_dummy_pkt_num != server_dummy_pkt_num else server_dummy_pkt_num
    
    first_incoming_pkt_time = trace[np.where(trace[:, 1] < 0)][0][0]
    last_pkt_time = trace[-1][0]
    
    client_timetable = getTimestamps(client_wnd, client_dummy_pkt)
    server_timetable = getTimestamps(server_wnd, server_dummy_pkt)
    server_timetable[:, 0] += first_incoming_pkt_time
    
    client_timetable = client_timetable[np.where(start_padding_time + client_timetable[:, 0] <= last_pkt_time)]
    server_timetable = server_timetable[np.where(start_padding_time + server_timetable[:, 0] <= last_pkt_time)]
    
    dummy_size = 512  # Assuming a default packet size for dummy packets
    client_pkts = np.column_stack((client_timetable, 888 * np.ones(len(client_timetable)), dummy_size * np.ones(len(client_timetable))))
    server_pkts = np.column_stack((server_timetable, -888 * np.ones(len(server_timetable)), dummy_size * np.ones(len(server_timetable))))
    
    noisy_trace = np.vstack((trace, client_pkts, server_pkts))
    noisy_trace = noisy_trace[noisy_trace[:, 0].argsort(kind="mergesort")]
    return noisy_trace

def getTimestamps(wnd, num):
    timestamps = sorted(np.random.rayleigh(wnd, num))
    return np.reshape(timestamps, (len(timestamps), 1))

if __name__ == "__main__":
    global client_dummy_pkt_num, server_dummy_pkt_num, client_min_dummy_pkt_num, server_min_dummy_pkt_num
    global max_wnd, min_wnd, start_padding_time
    
    args, config = parse_arguments()
    
    if not os.path.exists(args.p):
        raise FileNotFoundError(f"Path {args.p} not found")
    
    client_min_dummy_pkt_num = int(config.get("client_min_dummy_pkt_num", 1))
    server_min_dummy_pkt_num = int(config.get("server_min_dummy_pkt_num", 1))
    client_dummy_pkt_num = int(config.get("client_dummy_pkt_num", 300))
    server_dummy_pkt_num = int(config.get("server_dummy_pkt_num", 300))
    start_padding_time = int(config.get("start_padding_time", 0))
    max_wnd = float(config.get("max_wnd", 10))
    min_wnd = float(config.get("min_wnd", 10))
    
    flist = sorted([os.path.join(args.p, f) for f in os.listdir(args.p) if "-" in f])
    
    output_dir = init_directories(section=args.section)
    logger.info(f"Traces are dumped to {output_dir}")
    
    start = time.time()
    for i, f in tqdm(enumerate(flist), total=len(flist)):
        logger.debug(f"Simulating {f}")
        simulate(f)
    
    logger.info(f"Time: {time.time() - start}")
