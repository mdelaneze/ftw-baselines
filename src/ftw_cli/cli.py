import click

# Imports are in the functions below to speed-up CLI startup time
# Some of the ML related imports (presumable torch) are very slow
# See https://github.com/fieldsoftheworld/ftw-baselines/issues/40

@click.group()
def ftw():
    """Fields of The World (FTW) - Command Line Interface"""
    pass

## Data group

@ftw.group()
def data():
    """Downloading, unpacking, and preparing the FTW dataset."""
    pass

@data.command("download", help="Download the FTW dataset.")
@click.option('--out', '-o', type=str, default="./data", help="Folder where the files will be downloaded to. Defaults to './data'.")
@click.option('--clean_download', '-f', is_flag=True, help="If set, the script will delete the root folder before downloading.")
@click.option('--countries', type=str, default="all", help="Comma-separated list of countries to download. If 'all' (default) is passed, downloads all available countries.")
def data_download(out, clean_download, countries):
    from ftw_cli.download import download as fn
    fn(out, clean_download, countries)

@data.command("unpack", help="Unpack the downloaded FTW dataset. Specify the folder where the data is located via INPUT. Defaults to './data'.")
@click.argument('input', type=str, default="./data")
def data_unpack(input):
    from ftw_cli.unpack import unpack as fn
    fn(input)

data.add_command(data_download)
data.add_command(data_unpack)

### Model group

@ftw.group()
def model():
    """Training and testing FTW models."""
    pass

@model.command("fit", help="Fit the model")
@click.option('--config', '-c', required=True, type=click.Path(exists=True), help='Path to the config file (required)')
@click.option('--ckpt_path', type=click.Path(exists=True), help='Path to a checkpoint file to resume training from')
@click.argument('cli_args', nargs=-1, type=click.UNPROCESSED)  # Capture all remaining arguments
def model_fit(config, ckpt_path, cli_args):
    from ftw_cli.model import fit
    fit(config, ckpt_path, cli_args)

@model.command("test", help="Test the model")
@click.option('--model', '-m', required=True, type=str, help='Path to model checkpoint')
@click.option('--dir', type=str, default="data/ftw", help='Directory of dataset')
@click.option('--gpu', type=int, default=0, help='GPU to use')
@click.option('--countries', type=str, multiple=True, required=True, help='Countries to evaluate on')
@click.option('--postprocess', is_flag=True, help='Apply postprocessing to the model output')
@click.option('--iou_threshold', type=float, default=0.5, help='IoU threshold for matching predictions to ground truths')
@click.option('--out', '-o', type=str, default="metrics.json", help='Output file for metrics')
@click.option('--model_predicts_3_classes', is_flag=True, help='Whether the model predicts 3 classes or 2 classes')
@click.option('--test_on_3_classes', is_flag=True, help='Whether to test on 3 classes or 2 classes')
@click.option('--temporal_options', type=str, default="stacked", help='Temporal option (stacked, windowA, windowB, etc.)')
@click.argument('cli_args', nargs=-1, type=click.UNPROCESSED)  # Capture all remaining arguments
def model_test(model, dir, gpu, countries, postprocess, iou_threshold, out, model_predicts_3_classes, test_on_3_classes, temporal_options, cli_args):
    from ftw_cli.model import test
    test(model, dir, gpu, countries, postprocess, iou_threshold, out, model_predicts_3_classes, test_on_3_classes, temporal_options, cli_args)

model.add_command(model_fit)
model.add_command(model_test)

### Inference group

WIN_HELP = "URL to or Microsoft Planetary Computer ID of an Sentinel-2 L2A STAC item for the window {x} image"

@ftw.group()
def inference():
    """Inference-related commands."""
    pass

@inference.command("download", help="Download 2 Sentinel-2 scenes & stack them in a single file for inference.")
@click.option('--win_a', type=str, required=True, help=WIN_HELP.format(x="A"))
@click.option('--win_b', type=str, required=True, help=WIN_HELP.format(x="B"))
@click.option('--out', '-o', type=str, required=True, help="Filename to save results to")
@click.option('--overwrite', '-f', is_flag=True, help="Overwrites the outputs if they exist")
def inference_download(win_a, win_b, out, overwrite):
    from ftw_cli.inference import create_input
    create_input(win_a, win_b, out, overwrite)

@inference.command("run", help="Run inference on the stacked Sentinel-2 L2A satellite images specified via INPUT.")
@click.argument('input', type=click.Path(exists=True), required=True)
@click.option('--model', '-m', type=click.Path(exists=True), required=True, help="Path to the model checkpoint.")
@click.option('--out', '-o', type=str, required=True, help="Output filename.")
@click.option('--resize_factor', type=int, default=2, help="Resize factor to use for inference.")
@click.option('--gpu', type=int, help="GPU ID to use. If not provided, CPU will be used by default.")
@click.option('--patch_size', type=int, default=1024, help="Size of patch to use for inference.")
@click.option('--batch_size', type=int, default=2, help="Batch size.")
@click.option('--padding', type=int, default=64, help="Pixels to discard from each side of the patch.")
@click.option('--overwrite', '-f', is_flag=True, help="Overwrite outputs if they exist.")
@click.option('--mps_mode', is_flag=True, help="Run inference in MPS mode (Apple GPUs).")
def inference_run(input, model, out, resize_factor, gpu, patch_size, batch_size, padding, overwrite, mps_mode):
    from ftw_cli.inference import run
    run(input, model, out, resize_factor, gpu, patch_size, batch_size, padding, overwrite, mps_mode)

@inference.command("polygonize", help="Polygonize the output from inference for the raster image given via INPUT.")
@click.argument('input', type=click.Path(exists=True), required=True)
@click.option('--out', '-o', type=str, required=True, help="Output filename for the polygonized data.")
@click.option('--simplify', type=float, default=None, help="Simplification factor to use when polygonizing.")
@click.option('--min_size', type=float, default=500, help="Minimum area size in square meters to include in the output.")
@click.option('--overwrite', '-f', is_flag=True, help="Overwrite outputs if they exist.")
def inference_polygonize(input, out, simplify, min_size, overwrite):
    from ftw_cli.inference import polygonize
    polygonize(input, out, simplify, min_size, overwrite)

inference.add_command(inference_download)
inference.add_command(inference_polygonize)
inference.add_command(inference_run)

if __name__ == "__main__":
    ftw()
