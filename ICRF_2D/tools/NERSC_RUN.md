# NERSC run

There are only two commands to remember.

## Login once

```bash
tools/nersc_login_once.sh
```

This creates the temporary NERSC SSH key/certificate, usually valid for about
24 hours.

Double-click equivalent:

```text
tools/nersc_login_once.command
```

## Run and fetch

```bash
tools/run_nersc_and_fetch.sh
```

This syncs the project to Perlmutter, submits the Slurm job, waits for it to
finish, prints the final job status, and fetches `output_data/`, `figures/`,
and `logs/` back to this folder.

Double-click equivalent:

```text
tools/run_nersc_and_fetch.command
```
