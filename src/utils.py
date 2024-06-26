class IdMapper:
    def __init__(self, coco_dataset: dict):
        self.map_img = {x['id']: x['sly_id'] for x in coco_dataset['images']}
        self.map_obj = {x['id']: x['sly_id'] for x in coco_dataset['annotations']}