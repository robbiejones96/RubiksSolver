from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.train import RMSPropOptimizer

def buildModel(inputSize):
	cubeInputs = Input(shape=(inputSize,), name="CubeInputs")
	firstLayer = Dense(4096, activation="elu", name="FirstLayer")(cubeInputs)
	secondLayer = Dense(2048, activation="elu", name="SecondLayer")(firstLayer)
	thirdLayerValues = Dense(512, activation="elu", name="ThirdLayerValues")(secondLayer)
	thirdLayerPolicies = Dense(512, activation="elu", name="ThirdLayerPolicies")(secondLayer)
	valueOutput = Dense(1, name="ValueOutput")(thirdLayerValues)
	policyOutput = Dense(12, activation="softmax", name="PolicyOutput")(thirdLayerPolicies)
	model = Model(inputs=cubeInputs, outputs=[valueOutput, policyOutput])
	return model

def compileModel(model, learningRate):
	model.compile(optimizer=RMSPropOptimizer(learning_rate=learningRate),
				  loss={"ValueOutput" : "mean_squared_error",
				        "PolicyOutput" : "sparse_categorical_crossentropy"})