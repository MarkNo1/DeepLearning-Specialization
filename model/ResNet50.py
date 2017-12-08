from ResNetModel import ResNet

# Example of usage..

model = ResNet.model_50(input_shape=(64, 64, 3), classes=6)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=2, batch_size=32)
