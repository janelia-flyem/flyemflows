import sys
from confiddler import load_config
from neuclease.util import Grid
from flyemflows.volumes import VolumeService, GrayscaleVolumeSchema
from flyemflows.brick.brickwall import BrickWall


def convert_grayscale(config_path, client=None):
    """
    Simple example showing how to:
     - create an input service (agnostic to data format)
     - read it into a distributed array (BrickWall)
     - realign it to an output array
     - write the realigned data (agnostic to format)
    
    The input will be accessed according to it's preferred access pattern,
    and the output will be written according to it's preferreed access pattern
    (e.g. entire slices if reading from a PNG stack, or blocks if using N5.)

    Caveats:
     
     - This does not implement a Workflow subclass
       (though there isn't much more to it).
     
     - For simplicity, this code assumes that the entire volume can be loaded
       into your cluster's RAM.  For large volumes, that won't work.
       A more robust solution would split the input volume into large
       "slabs" and process them each in turn.

    Example:

        # Set up some input data
        from flyemflows.util.n5 import export_to_multiscale_n5
        volume = np.random.randint(255, size=(500,500,500), dtype=np.uint8)
        export_to_multiscale_n5(volume, '/tmp/test-vol.n5')

        # Write the config file:
        cat < /tmp/test-config.yaml
        input:
          n5:
            path: /tmp/test-vol.n5
            dataset: 's0'

        output:
          slice-files:
            slice-path-format: '/tmp/test-slices/z{:04}.png'

        # Run this script:
        python convert_grayscale.py /tmp/test-config.yaml

    """
    # Define the config file schema
    schema = {
        "properties": {
            "input": GrayscaleVolumeSchema,
            "output": GrayscaleVolumeSchema
        }
    }
    
    # Load config (injects defaults for missing values)
    config = load_config(config_path, schema)

    # Create input service and input 'bricks'
    input_svc = VolumeService.create_from_config(config["input"])
    input_wall = BrickWall.from_volume_service(input_svc, client=client)

    # Copy bounding box from input to output
    config["output"]["geometry"]["bounding-box"] = config["input"]["geometry"]["bounding-box"]

    # Create output service and redistribute
    # data using the output's preferred grid
    output_svc = VolumeService.create_from_config(config["output"])
    output_grid = Grid(output_svc.preferred_message_shape)
    output_wall = input_wall.realign_to_new_grid(output_grid)

    # Write the data: one task per output brick
    # (e.g. output slices, if exporting to PNGs)
    def write_brick(brick):
        output_svc.write_subvolume(brick.volume, brick.physical_box[0])
    output_wall.bricks.map(write_brick).compute()

    print(f"DONE exporting")

if __name__ == "__main__":
    # Usage:
    #  python convert_grayscale.py my-config.yaml 
    convert_grayscale(sys.argv[1])
