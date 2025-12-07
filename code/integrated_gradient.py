import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
from code import *
from tqdm import tqdm


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


train_filepath = r'dataset/train_dataset.csv'
train_seqs, y_train = load_dataset(train_filepath)
protein_bert_train = extract_embedding_features(train_seqs)
BLOSUM62_train = BLOSUM62(train_seqs)
one_hot_train = np.array(train_seqs).astype(np.float32)
model_1 = CNN(protein_bert_train)
model_2 = BiGRU()
model_3 = CNN(BLOSUM62_train)
model_atnn = ensemble_model()


def integrate_model():
    combined = tf.concat([model_1.output, model_2.output, model_3.output], axis=-1)
    output = model_atnn(combined)
    model = Model(inputs=[model_1.input, model_2.layers[2].input, model_3.input], outputs=[output])
    return model


model = integrate_model()


def get_array(x, y, z):
    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)
    z = np.expand_dims(z, axis=0)
    return x, y, z


def get_gradients(x, y, z, top_pred_idx):
    x, y, z = get_array(x, y, z)
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    z = tf.cast(z, tf.float32)
    inputs = [x, y, z]

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds = model(inputs)
        probs = preds[:, top_pred_idx]

    grads = tape.gradient(probs, inputs)
    return grads


def get_integrated_gradients(inputs, top_pred_idx=0, baseline=None, num_steps=50):
    if baseline is None:
        baseline = [np.zeros_like(inputs[i]).astype(np.float32) for i in range(len(inputs))]
    else:
        baseline = [i.astype(np.float32) for i in baseline]

    # 1. Do interpolation.
    interpolated_inputs = []
    for step in range(50 + 1):
        interpolated_input = []
        for i in range(len(inputs)):
            interpolated_values = baseline[i] + (step / num_steps) * (inputs[i] - baseline[i])
            interpolated_input.append(interpolated_values)
        interpolated_inputs.append(interpolated_input)

    # 2. Get the gradients
    grads = [list() for i in range(3)]
    for i, interpolated_input in enumerate(interpolated_inputs):
        x, y, z = interpolated_input
        grad = get_gradients(x, y, z, top_pred_idx=top_pred_idx)
        for i in range(len(inputs)):
            grads[i].append(grad[i])
    for i in range(len(grads)):
        grads[i] = tf.convert_to_tensor(grads[i], dtype=tf.float32)

    avg_grads = []
    # 3. Approximate the integral using the trapezoidal rule
    for i in range(len(grads)):
        grad = grads[i]
        grad = (grad[1:] + grad[:-1]) / 2.0
        avg_grad = tf.reduce_mean(grad, axis=0)
        avg_grads.append(avg_grad)

    # 4. Calculate integrated gradients and return
    integrated_grads = []
    for i in range(len(avg_grads)):
        integrated_grad = (inputs[i] - baseline[i]) * avg_grads[i]
        integrated_grads.append(integrated_grad)
    return integrated_grads


def summarize_attributions_sample_level(grads):
    attrs = []
    for i in range(len(grads)):
        grad = grads[i]
        grad = tf.squeeze(grad)
        grad = tf.reduce_sum(grad, axis=1) / tf.norm(grad)
        attrs.append(grad)
    return attrs


def calculate_global_attributions(pbf, onehot_f, BLOSUM62_f):
    attrs = []
    n_samples = pbf.shape[0]
    with tqdm(total=n_samples) as pbar:
        for x, y, z in zip(pbf, onehot_f, BLOSUM62_f):
            y = model_2.layers[1](y)
            inputs = [x, y, z]
            ig = get_integrated_gradients(inputs)
            attr = summarize_attributions_sample_level(ig)
            attrs.append(attr)
            pbar.update(1)
    attrs = np.array(attrs)
    np.save('./interpretability_analysis/attrs.npy', attrs)
    return attrs


if __name__ == '__main__':
    attrs = calculate_global_attributions(protein_bert_train, one_hot_train, BLOSUM62_train)
