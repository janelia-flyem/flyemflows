from . import NewAxisOrderSchema, RescaleLevelSchema, LabelMapSchema

GrayscaleAdapters = {
    "description": "Adapter properties that can be applied to a grayscale service\n",
    "default" : {},
    "properties": {
        "transpose-axes": NewAxisOrderSchema,
        "rescale-level": RescaleLevelSchema
    }
}

SegmentationAdapters = {
    "description": "Adapter properties that can be applied to a grayscale service\n",
    "default" : {},
    "properties": {
        "transpose-axes": NewAxisOrderSchema,
        "rescale-level": RescaleLevelSchema,
        "apply-labelmap": LabelMapSchema
    }
}
