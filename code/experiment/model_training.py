from code import *
from cross_validation import load_dataset, mkdir
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy as np
import os


def train_model(model, model_name, x_train, y, x_valid, y_valid, outputs):
    filepath = os.path.join(outputs, model_name)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=45)
    best_saving = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', mode='auto',
                                                       verbose=0, save_best_only=True, save_weights_only=True)
    model.fit(x_train, y, validation_data=(x_valid, y_valid), epochs=300, batch_size=32, shuffle=True,
              callbacks=[early_stopping, best_saving], verbose=0)
    model.load_weights(filepath)
    return model


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    os.chdir('../../')
    # 1.load_data
    train_filepath = r'dataset/train_dataset.csv'
    test_filepath = r'dataset/test_dataset.csv'
    train_seqs, y_train = load_dataset(train_filepath)
    test_seqs, y_test = load_dataset(test_filepath)

    # 2.extrac features
    protein_bert_train = extract_embedding_features(train_seqs)
    protein_bert_test = extract_embedding_features(test_seqs)
    tf.keras.backend.clear_session()

    BLOSUM62_train = BLOSUM62(train_seqs)
    BLOSUM62_test = BLOSUM62(test_seqs)

    one_hot_train = np.array(to_embedding_numeric(train_seqs)).astype(np.float32)
    one_hot_test = np.array(to_embedding_numeric(test_seqs)).astype(np.float32)

    # 3.train model
    folds = StratifiedKFold(20, shuffle=True, random_state=42).split(protein_bert_train, y_train)
    # 5% for validation to avoid model overfitting
    train_index, valid_index = next(folds)
    train_x_emb, train_y = protein_bert_train[train_index], y_train[train_index]
    valid_x_emb, valid_y = protein_bert_train[valid_index], y_train[valid_index]
    train_x_onehot, valid_x_onehot = one_hot_train[train_index], one_hot_train[valid_index]
    train_x_BLOSUM62, valid_x_BLOSUM62 = BLOSUM62_train[train_index], BLOSUM62_train[valid_index]

    outputs = r'model/dedl_kcr/'
    mkdir(outputs)
    modelName = 'model1.h5'
    network_1 = CNN(train_x_emb)
    network_1 = train_model(network_1, modelName, train_x_emb, train_y, valid_x_emb, valid_y, outputs)

    modelName = 'model2.h5'
    network_2 = BiGRU()
    network_2 = train_model(network_2, modelName, train_x_onehot, train_y, valid_x_onehot, valid_y, outputs)

    modelName = 'model3.h5'
    network_3 = CNN(train_x_BLOSUM62)
    network_3 = train_model(network_3, modelName, train_x_BLOSUM62, train_y, valid_x_BLOSUM62, valid_y, outputs)

    att_model = ensemble_model()
    att_train_input = np.concatenate((network_1.predict(train_x_emb, verbose=0),
                                      network_2.predict(train_x_onehot, verbose=0),
                                      network_3.predict(train_x_BLOSUM62, verbose=0)), axis=-1)
    att_valid_input = np.concatenate((network_1.predict(valid_x_emb, verbose=0),
                                      network_2.predict(valid_x_onehot, verbose=0),
                                      network_3.predict(valid_x_BLOSUM62, verbose=0)), axis=-1)
    modelName = 'model_ensemble.h5'
    filepath_att = os.path.join(outputs, modelName)
    att_model = train_model(att_model, modelName, att_train_input, train_y, att_valid_input, valid_y, outputs)