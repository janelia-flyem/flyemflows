from . import NewAxisOrderSchema, TranslateSchema, RescaleLevelSchema, LabelMapSchema

ADAPTER_ORDER = ['apply-labelmap', 'transpose-axes', 'translate', 'rescale-level']

GrayscaleAdapters = {
    "description": "Adapter properties that can be applied to a grayscale service\n",
    "default" : {},
    "properties": {
        "transpose-axes": NewAxisOrderSchema,
        "translate": TranslateSchema,
        "rescale-level": RescaleLevelSchema
    }
}

# Ensure config entries are listed in the canonical order.
_gray_adapters = GrayscaleAdapters['properties'].keys()
assert list(_gray_adapters) == list(filter(lambda x: x in _gray_adapters, ADAPTER_ORDER))

SegmentationAdapters = {
    "description": "Adapter properties that can be applied to a grayscale service\n",
    "default" : {},
    "properties": {
        "apply-labelmap": LabelMapSchema,
        "transpose-axes": NewAxisOrderSchema,
        "translate": TranslateSchema,
        "rescale-level": RescaleLevelSchema
    }
}

# Ensure config entries are listed in the canonical order.
_seg_adapters = SegmentationAdapters['properties'].keys()
assert list(_seg_adapters) == list(filter(lambda x: x in _seg_adapters, ADAPTER_ORDER))
