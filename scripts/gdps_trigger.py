#!/usr/bin/env python3
"""
GDPS quasi-real-time trigger script.

Queries InfluxDB for the latest sensor readings, uploads as sensor CSV
to Galaxy, then invokes the gdps_realtime_transient workflow.

Run on a schedule (e.g. cron, systemd timer, or via /loop):
    python3 gdps_trigger.py --config trigger_config.json
"""

import argparse
import csv
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(state_file, state):
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def query_influxdb(influx_cfg, t0, dtim, dry_run=False):
    from influxdb_client import InfluxDBClient

    client = InfluxDBClient(
        url=influx_cfg['url'],
        token=influx_cfg['token'],
        org=influx_cfg['org'],
    )
    query_api = client.query_api()

    window = influx_cfg['window_seconds']
    measurement = influx_cfg['measurement']
    bucket = influx_cfg['bucket']
    fields = influx_cfg['fields']

    field_filter = ' or '.join(f'r["_field"] == "{f}"' for f in fields.values())
    flux_query = f"""
from(bucket: "{bucket}")
  |> range(start: -{window}s)
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => {field_filter})
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""

    if dry_run:
        print("[dry-run] Flux query:")
        print(flux_query)

    tables = query_api.query(flux_query)
    client.close()

    t0_dt = datetime.fromisoformat(t0.replace('Z', '+00:00'))

    rows = []
    for table in tables:
        for record in table.records:
            ts = record.get_time()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            elapsed_raw = (ts - t0_dt).total_seconds()
            step = round(elapsed_raw / dtim)
            elapsed_seconds = step * dtim

            row = {'elapsed_seconds': elapsed_seconds}
            for col_name, field_name in fields.items():
                val = record.values.get(field_name)
                if val is not None:
                    row[col_name] = val
            rows.append(row)

    return rows


def write_csv(rows, fields):
    col_names = list(fields.keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=['elapsed_seconds'] + col_names,
                            extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def upload_csv_to_galaxy(gi, history_id, csv_content, dry_run=False):
    if dry_run:
        print("[dry-run] Would upload CSV to Galaxy history:", history_id)
        print(csv_content[:500])
        return "dry-run-dataset-id"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        tmp_path = f.name

    try:
        result = gi.tools.upload_file(tmp_path, history_id, file_type='csv')
        dataset_id = result['outputs'][0]['id']
        print(f"Uploaded sensor CSV, dataset ID: {dataset_id}")
        return dataset_id
    finally:
        os.unlink(tmp_path)


def build_workflow_inputs(galaxy_cfg, sensor_csv_id, checkpoint_id):
    input_map = {}
    for label, dataset_id in galaxy_cfg['input_map'].items():
        input_map[label] = {'id': dataset_id, 'src': 'hda'}
    input_map['sensor_csv'] = {'id': sensor_csv_id, 'src': 'hda'}
    input_map['chk_file'] = {'id': checkpoint_id, 'src': 'hda'}
    return input_map


def invoke_workflow(gi, galaxy_cfg, workflow_inputs, dry_run=False):
    if dry_run:
        print("[dry-run] Would invoke workflow:", galaxy_cfg['workflow_id'])
        print("Inputs:", json.dumps(workflow_inputs, indent=2))
        return "dry-run-invocation-id"

    result = gi.workflows.invoke_workflow(
        workflow_id=galaxy_cfg['workflow_id'],
        history_id=galaxy_cfg['history_id'],
        inputs=workflow_inputs,
    )
    invocation_id = result['id']
    print(f"Workflow invoked, invocation ID: {invocation_id}")
    return invocation_id


def poll_for_completion(gi, invocation_id, timeout_seconds=3600):
    import time
    print(f"Polling for completion of invocation {invocation_id}...")
    elapsed = 0
    poll_interval = 30
    while elapsed < timeout_seconds:
        invocation = gi.invocations.show_invocation(invocation_id)
        state = invocation.get('state', 'unknown')
        print(f"  State: {state} (elapsed {elapsed}s)")
        if state in ('scheduled', 'ok'):
            return invocation
        if state in ('error', 'failed', 'cancelled'):
            print(f"Workflow ended with state: {state}")
            return invocation
        time.sleep(poll_interval)
        elapsed += poll_interval
    print(f"Timed out after {timeout_seconds}s")
    return None


def find_checkpoint_dataset(gi, history_id, invocation_id):
    invocation = gi.invocations.show_invocation(invocation_id)
    for step in invocation.get('steps', []):
        for job_id in step.get('job_ids', []):
            job = gi.jobs.show_job(job_id, full_details=True)
            for output_name, output_info in job.get('outputs', {}).items():
                if 'chk' in output_name.lower() or output_name == 'output_chk':
                    return output_info.get('id')
    return None


def main():
    parser = argparse.ArgumentParser(description='GDPS quasi-real-time trigger')
    parser.add_argument('--config', required=True, help='Path to trigger_config.json')
    parser.add_argument('--state-file', default='gdps_trigger_state.json',
                        help='Path to state file for persisting checkpoint dataset ID')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print CSV and workflow inputs without uploading or invoking')
    parser.add_argument('--poll', action='store_true',
                        help='Poll for completion and update checkpoint in state file')
    args = parser.parse_args()

    config = load_config(args.config)
    state = load_state(args.state_file)

    influx_cfg = config['influx']
    galaxy_cfg = config['galaxy']
    sim_cfg = config['sim']

    dtim = sim_cfg['dtim']
    t0 = sim_cfg['t0']

    print(f"Querying InfluxDB for last {influx_cfg['window_seconds']}s of data...")
    rows = query_influxdb(influx_cfg, t0, dtim, dry_run=args.dry_run)

    if not rows:
        print("No sensor data in the query window. Exiting without invoking workflow.")
        sys.exit(0)

    print(f"Got {len(rows)} rows from InfluxDB.")

    csv_content = write_csv(rows, influx_cfg['fields'])

    if args.dry_run:
        print("[dry-run] CSV content:")
        print(csv_content)

    from bioblend.galaxy import GalaxyInstance
    gi = GalaxyInstance(url=galaxy_cfg['url'], key=galaxy_cfg['api_key'])

    sensor_csv_id = upload_csv_to_galaxy(gi, galaxy_cfg['history_id'], csv_content,
                                         dry_run=args.dry_run)

    checkpoint_id = state.get('checkpoint_dataset_id', galaxy_cfg.get('checkpoint_dataset_id'))
    if not checkpoint_id:
        print("Error: no checkpoint_dataset_id in state file or config.")
        sys.exit(1)

    workflow_inputs = build_workflow_inputs(galaxy_cfg, sensor_csv_id, checkpoint_id)

    invocation_id = invoke_workflow(gi, galaxy_cfg, workflow_inputs, dry_run=args.dry_run)

    print(f"Invocation ID: {invocation_id}")

    if args.poll and not args.dry_run:
        invocation = poll_for_completion(gi, invocation_id)
        if invocation and invocation.get('state') in ('scheduled', 'ok'):
            new_chk_id = find_checkpoint_dataset(gi, galaxy_cfg['history_id'], invocation_id)
            if new_chk_id:
                state['checkpoint_dataset_id'] = new_chk_id
                save_state(args.state_file, state)
                print(f"Updated checkpoint dataset ID to: {new_chk_id}")
            else:
                print("Could not find new checkpoint dataset in invocation outputs.")


if __name__ == '__main__':
    main()
