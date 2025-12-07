from code import *
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import os


def train_model(model, model_name, x_train, y, x_valid, y_valid, outputs):
    filepath = os.path.join(outputs, model_name)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=45)
    best_saving = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', mode='auto',
                                                       verbose=0, save_best_only=True, save_weights_only=True)
    model.fit(x_train, y, validation_data=(x_valid, y_valid), epochs=300, batch_size=32, shuffle=True,
              callbacks=[early_stopping, best_saving], verbose=0)
    model.load_weights(filepath)
    p_1 = model.predict(x_valid, verbose=0)
    tmp_result = np.zeros((len(y_valid), 3))
    tmp_result[:, 0], tmp_result[:, 1:] = y_valid, p_1
    return tmp_result


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


    os.chdir('../../')
    print(os.getcwd())
    # 1.load_data
    train_filepath = r'dataset/train_dataset.csv'
    train_seqs, y_train = load_dataset(train_filepath)

    # 2.extrac features
    protein_bert_train = extract_embedding_features(train_seqs)
    tf.keras.backend.clear_session()

    BLOSUM62_train = BLOSUM62(train_seqs)

    one_hot_train = np.array(to_embedding_numeric(train_seqs)).astype(np.float32)

    # 20-fold
    outputs = r'code/experiment/results/cross-validation/'
    mkdir(outputs)
    folds = StratifiedKFold(20, shuffle=True, random_state=42).split(protein_bert_train, y_train)
    prediction_result_cv = []
    prediction_result_cv_pb = []
    prediction_result_cv_oh = []
    prediction_result_cv_bl = []
    for i, (train, valid) in enumerate(folds):
        train_x_emb, train_y = protein_bert_train[train], y_train[train]
        valid_x_emb, valid_y = protein_bert_train[valid], y_train[valid]
        train_x_onehot, valid_x_onehot = one_hot_train[train], one_hot_train[valid]
        train_x_BLOSUM62, valid_x_BLOSUM62 = BLOSUM62_train[train], BLOSUM62_train[valid]
        modelName = 'model1_' + str(i + 1) + '.h5'
        network_1 = CNN(train_x_emb)
        tmp_result = train_model(network_1, modelName, train_x_emb, train_y, valid_x_emb, valid_y, outputs)
        prediction_result_cv_pb.append(tmp_result)

        modelName = 'model2_' + str(i + 1) + '.h5'
        network_2 = BiGRU()
        tmp_result = train_model(network_2, modelName, train_x_onehot, train_y, valid_x_onehot, valid_y, outputs)
        prediction_result_cv_oh.append(tmp_result)

        modelName = 'model3_' + str(i + 1) + '.h5'
        network_3 = CNN(train_x_BLOSUM62)
        tmp_result = train_model(network_3, modelName, train_x_BLOSUM62, train_y, valid_x_BLOSUM62, valid_y, outputs)
        prediction_result_cv_bl.append(tmp_result)

        att_model = ensemble_model()
        att_train_input = np.concatenate((network_1.predict(train_x_emb, verbose=0),
                                          network_2.predict(train_x_onehot, verbose=0),
                                          network_3.predict(train_x_BLOSUM62, verbose=0)), axis=-1)
        att_valid_input = np.concatenate((network_1.predict(valid_x_emb, verbose=0),
                                          network_2.predict(valid_x_onehot, verbose=0),
                                          network_3.predict(valid_x_BLOSUM62, verbose=0)), axis=-1)
        modelName = 'model_ensemble_' + str(i + 1) + '.h5'
        filepath_att = os.path.join(outputs, modelName)
        tmp_result = train_model(att_model, modelName, att_train_input, train_y, att_valid_input, valid_y, outputs)
        prediction_result_cv.append(tmp_result)

    save_val_result(prediction_result_cv, outputs, "att_ensemble")
    save_val_result(prediction_result_cv_pb, outputs, "protein_bert")
    save_val_result(prediction_result_cv_oh, outputs, "one_hot")
    save_val_result(prediction_result_cv_bl, outputs, "blosum62")
