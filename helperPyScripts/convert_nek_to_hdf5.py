#!/usr/bin/env pvpython
"""Convert a Nek5000 time series to VTK HDF using ParaView.

Run this script with ParaView's Python interpreter, for example:

    pvpython convert_nek_to_hdf5.py case.nek5000
    pvpython convert_nek_to_hdf5.py case.nek5000 custom_name
    pvpython convert_nek_to_hdf5.py --single-file case.nek5000

The ``.nek5000`` file is the small descriptor file that refers to the Nek
field files.  All point-data arrays exposed by ParaView's Nek5000 reader are
written.  By default, each timestep is written to a separate VTK HDF file;
``--single-file`` stores all timesteps in one temporal VTK HDF file instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Nek5000 time series to HDF5-backed VTK HDF."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Nek5000 descriptor file (normally *.nek5000)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help=(
            "optional output name or path. The .vtkhdf extension is added "
            "automatically; defaults to the input descriptor's name"
        ),
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help=(
            "store every Nek timestep in one temporal VTK HDF file instead "
            "of writing one file per timestep"
        ),
    )
    return parser.parse_args(argv)


def validate_paths(
    input_path: Path, output_path: Path | None
) -> tuple[Path, Path]:
    input_path = input_path.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Nek5000 descriptor not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".vtkhdf")
    else:
        output_path = output_path.expanduser()
        if output_path.suffix.lower() != ".vtkhdf":
            output_path = Path(f"{output_path}.vtkhdf")
        output_path = output_path.resolve()

    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_path.parent}"
        )

    return input_path, output_path


def convert(
    input_path: Path, output_path: Path, split_timesteps: bool = True
) -> list[Path]:
    try:
        from paraview.simple import Nek5000Reader, SaveData
    except ImportError as exc:
        raise RuntimeError(
            "ParaView's Python modules are unavailable. Run this script with "
            "pvpython (or pvbatch), not the system Python interpreter."
        ) from exc

    reader = Nek5000Reader(FileName=str(input_path))
    reader.UpdatePipelineInformation()

    # The reader discovers the arrays from the descriptor and field files.
    reader.PointArrays = list(reader.PointArrays.Available)
    reader.UpdatePipeline()

    if not split_timesteps:
        SaveData(
            str(output_path),
            proxy=reader,
            WriteAllTimeSteps=1,
        )
        return [output_path]

    timesteps = list(reader.TimestepValues)
    if not timesteps:
        raise RuntimeError("The Nek5000 reader reported no timesteps.")

    output_files = []
    for index, time_value in enumerate(timesteps):
        timestep_output = output_path.with_name(
            f"{output_path.stem}_t{index:05d}{output_path.suffix}"
        )
        reader.UpdatePipeline(time=time_value)
        SaveData(
            str(timestep_output),
            proxy=reader,
            WriteAllTimeSteps=0,
        )
        output_files.append(timestep_output)
        print(
            f"Saved timestep {index + 1}/{len(timesteps)} "
            f"(time={time_value:g}) to {timestep_output}"
        )

    return output_files


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        input_path, output_path = validate_paths(args.input, args.output)
        output_files = convert(
            input_path,
            output_path,
            split_timesteps=not args.single_file,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.single_file:
        print(f"Saved {output_files[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
