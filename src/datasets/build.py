from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""

SPLITTER_REGISTRY = Registry("DATASET")
SPLITTER_REGISTRY.__doc__ = """
Registry for funcctions to get list of patientsand nb of eyes per patients.

The registered object will be called with `obj(args)`.
The call should return a List of patients, and a List of nb of eyes associated.
"""

def splitter(args):
    """
    Create splitter helper and returns the list of patient and the associated number of eyes, 
    """

    name = args.dataset_name.capitalize() + "Splitter"
    list_patients, nb_eyes = SPLITTER_REGISTRY.get(name)(args)
    return list_patients, nb_eyes

def build_dataset(args, patient_list, transforms=None, random_sampling=True):
    """
    Build a dataset, defined by args and patient list.
    """
    name = args.dataset_name.capitalize() + "Dataset"
    dataset = DATASET_REGISTRY.get(name)(args, patient_list, transforms, random_sampling)
    return dataset