import pandas as pd


def _get_categories_categorized(data) -> pd.DataFrame:
    counts = {}
    for ix in range(len(data)):
        category = data.get_category(ix)
        try:
            counts[category] += 1
        except KeyError:
            counts[category] = 1
    return pd.DataFrame(dict(
        category_index=i,
        category_name=name,
        category_num_samples=counts[name]
    ) for i, name in enumerate(sorted(list(counts.keys()))))


def _get_categories_annotated(data) -> pd.DataFrame:
    num_total = {}
    num_samples = {}
    for ix in range(len(data)):
        annotations = data.get_annotations(ix)
        categories = annotations[data.get_category_name_column(annotations.columns)]
        for category in categories:
            try:
                num_total[category] += 1
            except KeyError:
                num_total[category] = 1
        for category in set(categories):
            try:
                num_samples[category] += 1
            except KeyError:
                num_samples[category] = 1
    return pd.DataFrame(dict(
        category_index=i,
        category_name=name,
        category_num_samples=num_samples[name],
        category_num_instances=num_total[name]
    ) for i, name in enumerate(sorted(list(num_total.keys()))))


def get_categories(data) -> pd.DataFrame:
    try:
        return _get_categories_categorized(data)
    except AttributeError:
        return _get_categories_annotated(data)
