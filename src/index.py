import getData
import generateFeature
import selectFeature
import trainModel

getData.getData()
generateFeature.generateFeature()

variables = selectFeature.selectFeature()
trainModel.trainModel(variables['X_train'], variables['X_test'], variables['y_train'], variables['y_test'])

