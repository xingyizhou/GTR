import json
import sys

LVIS_CAT_PATH = 'datasets/lvis/lvis_v1_val.json'

if __name__ == '__main__':
    print('Loading tao')
    tao_v05_path = sys.argv[1]
    data = json.load(open(tao_v05_path, 'r'))
    print('Loading LVIS cat')
    lvis_v1_cat = json.load(open(LVIS_CAT_PATH, 'r'))['categories']
    v05_cat = data['categories']
    v05id2cat = {x['id']: x for x in data['categories']}
    synset2v1 = {x['synset']: x for x in lvis_v1_cat}
    appeared_cat_id = set(x['category_id'] for x in data['annotations'])
    idmap = {}
    for cat_id in appeared_cat_id:
        cat = v05id2cat[cat_id]
        synset = cat['synset']
        if synset in synset2v1:
            v1cat = synset2v1[synset]
            idmap[cat_id] = v1cat['id']
        else:
            print('Not in v1!', cat)
    print('Ori cat', len(appeared_cat_id))
    print('New cat', len(idmap))

    new_anns = []
    drop = 0
    inst_cont = {y: 0 for x, y in idmap.items()}
    image_count = {y: set() for x, y in idmap.items()}
    for x in data['annotations']:
        if x['category_id'] in idmap:
            new_id = idmap[x['category_id']]
            x['category_id'] = new_id
            new_anns.append(x)
            inst_cont[new_id] += 1
            image_count[new_id].add(x['image_id'])
        else:
            drop = drop + 1

    print('Ori instances', len(data['annotations']))
    print('Drop instances', drop)
    print('New instances', len(new_anns))

    data['categories'] = lvis_v1_cat
    for x in data['categories']:
        if x['id'] in inst_cont.keys():
            x['image_count'] = len(image_count[x['id']])
            x['instance_count'] = inst_cont[x['id']]
        else:
            x['image_count'] = 0
            x['instance_count'] = 0

    for x in data['videos']:
        x['neg_category_ids'] = [idmap[xx] for xx in x['neg_category_ids'] if xx in idmap]
        x['not_exhaustive_category_ids'] = [idmap[xx] for xx in x['not_exhaustive_category_ids'] if xx in idmap]

    tracks = []
    for x in data['tracks']:
        if x['category_id'] in idmap:
            x['category_id'] = idmap[x['category_id']]
            tracks.append(x)
    data['tracks'] = tracks

    id2video = {x['id']: x for x in data['videos']}

    data['annotations'] = new_anns
    for x in data['images']:
        video = id2video[x['video_id']]
        x['neg_category_ids'] = video['neg_category_ids']
        x['not_exhaustive_category_ids'] = video['not_exhaustive_category_ids']


    save_path = tao_v05_path[:-5] + '_v1.json'
    print('saving to', save_path)
    json.dump(data, open(save_path, 'w'))