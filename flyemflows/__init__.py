import warnings
## Don't show the following warning from within pandas:
## FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
warnings.filterwarnings("ignore", module=r"pandas\..*", category=FutureWarning)


# TODO:
# DVIDSparkServices had a lot of sophisticated configuration in its __init__ file.
# Some (or most) of it should be copied here.

