import turicreate as tc
import sys

def tc_trainer(sframe_path, model_name='model_skeri', filter_label=None, max_iter=None, batch_sz=None):
    data = tc.SFrame(sframe_path)
    data_train, _ = data.random_split(1, seed=542014)
    # Filter the data by name of image file (for datasets with class labels in image names)
    if filter_label:
        data_train = data.filter_by([name for name in data['name'] if filter_label in name], 'name')

    print('Turning on GPU.')
    tc.config.set_num_gpus(1)
    if max_iter and batch_sz:
        mlmodel = tc.object_detector.create(data_train,  max_iterations=max_iter, batch_size=batch_sz)
    elif max_iter:
        mlmodel = tc.object_detector.create(data_train, max_iterations=max_iter)
    else:
        mlmodel = tc.object_detector.create(data_train)
    mlmodel.save(model_name+'.model')
    mlmodel.evaluate(data)
    try:
        mlmodel.export_coreml(model_name+'.mlmodel')
    except RuntimeError:
        print('Did not succeed in exporting the gpu model to coreml.')

if __name__ == "__main__":
    if len(sys.argv) > 1: # and sys.argv[1] == '-s':
        sframe_path = sys.argv[1]
    else:
        sframe_path = '/Users/rcrabb/PycharmProjects/map_and_mlmodel/datasets/art/art.sframe'
    tc_trainer(sframe_path)