import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical

len_v = 1536 # 4096

def resh(v):
	v1 = np.zeros((len(v),len_v))
	for i in range(len(v)):
		v1[i] = v[i][0]
	return v1

v1 = np.load('airplane1.npy')
v2 = np.load('airplane2.npy')
v3 = np.load('airplane3.npy')
v4 = np.load('airplane4.npy')
v5 = np.load('airplane5.npy')
v6 = np.load('airplane6.npy')
v7 = np.load('airplane7.npy')
v8 = np.load('airplane8.npy')
v9 = np.load('airplane9.npy')
v10= np.load('airplane10.npy')
v11= np.load('airplane11.npy')
v12= np.load('airplane12.npy')
v13= np.load('airplane13.npy')
v14= np.load('airplane14.npy')
v15= np.load('airplane15.npy')

X_airplane_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13),axis=0)
X_airplane_2 = np.concatenate((v14,v15),axis=0)

X_airplane_1 = resh(X_airplane_1)
X_airplane_2 = resh(X_airplane_2)

print(X_airplane_1.shape)

v1 = np.load('ship1.npy')
v2 = np.load('ship2.npy')
v3 = np.load('ship3.npy')
v4 = np.load('ship4.npy')
v5 = np.load('ship5.npy')
v6 = np.load('ship6.npy')
v7 = np.load('ship7.npy')
v8 = np.load('ship8.npy')
v9 = np.load('ship9.npy')
v10= np.load('ship10.npy')
v11= np.load('ship11.npy')
v12= np.load('ship12.npy')
v13= np.load('ship13.npy')
v14= np.load('ship14.npy')
v15= np.load('ship15.npy')

X_ship_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13),axis=0)
X_ship_2 = np.concatenate((v14,v15),axis=0)

X_ship_1 = resh(X_ship_1)
X_ship_2 = resh(X_ship_2)

print(X_ship_1.shape)

#print(X_airplane_1.shape,X_airplane_2.shape)
#print(X_ship_1.shape,X_ship_2.shape)

v1 = np.load('cat1.npy')
v2 = np.load('cat2.npy')
v3 = np.load('cat3.npy')
v4 = np.load('cat4.npy')
v5 = np.load('cat5.npy')
v6 = np.load('cat6.npy')
v7 = np.load('cat7.npy')
v8 = np.load('cat8.npy')
v9 = np.load('cat9.npy')
v10= np.load('cat10.npy')
v11= np.load('cat11.npy')
v12= np.load('cat12.npy')
v13= np.load('cat13.npy')
v14= np.load('cat14.npy')
v15= np.load('cat15.npy')

X_cat_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13),axis=0)
X_cat_2 = np.concatenate((v14,v15),axis=0)

X_cat_1 = resh(X_cat_1)
X_cat_2 = resh(X_cat_2)

print(X_cat_1.shape)

#print(X_cat_1.shape)

v1 = np.load('horse1.npy')
v2 = np.load('horse2.npy')
v3 = np.load('horse3.npy')
v4 = np.load('horse4.npy')
v5 = np.load('horse5.npy')
v6 = np.load('horse6.npy')
v7 = np.load('horse7.npy')
v8 = np.load('horse8.npy')
v9 = np.load('horse9.npy')
v10= np.load('horse10.npy')
v11= np.load('horse11.npy')
v12A=np.load('horse12-A.npy')
v12B=np.load('horse12-B.npy')
v13= np.load('horse13.npy')
v14= np.load('horse14.npy')
v15= np.load('horse15.npy')

X_horse_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12A,v12B,v13),axis=0)
X_horse_2 = np.concatenate((v14,v15),axis=0)

X_horse_1 = resh(X_horse_1)
X_horse_2 = resh(X_horse_2)

print
print(v11.shape)
print(v12A.shape)
print(v12B.shape)
print(v13.shape)

print(X_horse_1.shape)

v1 = np.load('computer1.npy')
v2 = np.load('computer2.npy')
v3 = np.load('computer3.npy')
v4 = np.load('computer4.npy')
v5 = np.load('computer5.npy')
v6 = np.load('computer6.npy')
v7 = np.load('computer7.npy')
v8 = np.load('computer8.npy')
v9 = np.load('computer9.npy')
v10= np.load('computer10.npy')
v11= np.load('computer11.npy')
v12= np.load('computer12.npy')
v13= np.load('computer13.npy')
v14= np.load('computer14.npy')
v15= np.load('computer15.npy')

X_computer_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13),axis=0)
X_computer_2 = np.concatenate((v14,v15),axis=0)

X_computer_1 = resh(X_computer_1)
X_computer_2 = resh(X_computer_2)

print
print(v11.shape)
print(v12.shape)
print(v13.shape)
print(X_computer_1.shape)

v1 = np.load('desert1.npy')
v2 = np.load('desert2.npy')
v3 = np.load('desert3.npy')
v4 = np.load('desert4.npy')
v5 = np.load('desert5.npy')
v6 = np.load('desert6.npy')
v7 = np.load('desert7.npy')
v8 = np.load('desert8.npy')
v9 = np.load('desert9.npy')
v10= np.load('desert10.npy')
v11= np.load('desert11.npy')
v12= np.load('desert12.npy')
v13= np.load('desert13.npy')
v14= np.load('desert14.npy')
v15= np.load('desert15.npy')

X_desert_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13),axis=0)
X_desert_2 = np.concatenate((v14,v15),axis=0)

X_desert_1 = resh(X_desert_1)
X_desert_2 = resh(X_desert_2)

print(X_desert_1.shape)

v1 = np.load('fish1.npy')
v2 = np.load('fish2.npy')
v3 = np.load('fish3.npy')
v4 = np.load('fish4.npy')
v5 = np.load('fish5.npy')
v6 = np.load('fish6.npy')
v7 = np.load('fish7.npy')
v8 = np.load('fish8.npy')
v9 = np.load('fish9.npy')
v10= np.load('fish10.npy')
v11= np.load('fish11.npy')
v12= np.load('fish12.npy')
v13= np.load('fish13.npy')
v14= np.load('fish14.npy')
v15= np.load('fish15.npy')

X_fish_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13),axis=0)
X_fish_2 = np.concatenate((v14,v15),axis=0)

X_fish_1 = resh(X_fish_1)
X_fish_2 = resh(X_fish_2)

print(X_fish_1.shape)

v1 = np.load('flower1.npy')
v2 = np.load('flower2.npy')
v3 = np.load('flower3.npy')
v4 = np.load('flower4.npy')
v5 = np.load('flower5.npy')
v6 = np.load('flower6.npy')
v7 = np.load('flower7.npy')
v8 = np.load('flower8.npy')
v9 = np.load('flower9.npy')
v10= np.load('flower10.npy')
v11= np.load('flower11.npy')
v12A=np.load('flower12-A.npy')
v12B=np.load('flower12-B.npy')
v13= np.load('flower13.npy')
v14= np.load('flower14.npy')
v15= np.load('flower15.npy')

X_flower_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12A,v12B,v13),axis=0)
X_flower_2 = np.concatenate((v14,v15),axis=0)

X_flower_1 = resh(X_flower_1)
X_flower_2 = resh(X_flower_2)

print(X_flower_1.shape)

v1 = np.load('dog1.npy')
v2 = np.load('dog2.npy')
v3 = np.load('dog3.npy')
v4 = np.load('dog4.npy')
v5 = np.load('dog5.npy')
v6 = np.load('dog6.npy')
v7 = np.load('dog7.npy')
v8 = np.load('dog8.npy')
v9 = np.load('dog9.npy')
v10= np.load('dog10.npy')
v11= np.load('dog11.npy')
v12= np.load('dog12.npy')
v13= np.load('dog13.npy')
v14= np.load('dog14.npy')
v15= np.load('dog15.npy')

X_dog_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13),axis=0)
X_dog_2 = np.concatenate((v14,v15),axis=0)

X_dog_1 = resh(X_dog_1)
X_dog_2 = resh(X_dog_2)


print(X_dog_1.shape)

v1 = np.load('fire1.npy')
v2 = np.load('fire2.npy')
v3 = np.load('fire3.npy')
v4 = np.load('fire4.npy')
v5 = np.load('fire5.npy')
v6 = np.load('fire6.npy')
v7 = np.load('fire7.npy')
v8 = np.load('fire8.npy')
v9 = np.load('fire9.npy')
v10= np.load('fire10.npy')
v11= np.load('fire11.npy')
v12= np.load('fire12.npy')
v13= np.load('fire13.npy')
v14= np.load('fire14.npy')
v15= np.load('fire15.npy')

X_fire_1 = np.concatenate((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13),axis=0)
X_fire_2 = np.concatenate((v14,v15),axis=0)

X_fire_1 = resh(X_fire_1)
X_fire_2 = resh(X_fire_2)

print(X_fire_1.shape)


y_airplane_1 = np.zeros(2600)
y_airplane_2 = np.zeros(400)
y_ship_1 = np.ones(2600)
y_ship_2 = np.ones(400)
y_cat_1  = 2*np.ones(2600)
y_cat_2  = 2*np.ones(400)
y_horse_1  = 3*np.ones(2600)
y_horse_2  = 3*np.ones(400)
y_computer_1=4*np.ones(2600)
y_computer_2=4*np.ones(400)
y_desert_1 = 5*np.ones(2600)
y_desert_2 = 5*np.ones(400)
y_fish_1 = 6*np.ones(2600)
y_fish_2 = 6*np.ones(400)
y_flower_1= 7*np.ones(2600)
y_flower_2= 7*np.ones(400)
y_dog_1 = 8*np.ones(2600)
y_dog_2 = 8*np.ones(400)
y_fire_1 = 9*np.ones(2600)
y_fire_2 = 9*np.ones(400)

#print(X_dog_2.shape)

X_train = np.concatenate((X_airplane_1,X_ship_1,X_cat_1,X_horse_1,X_computer_1,X_desert_1,X_fish_1,X_flower_1,X_dog_1,X_fire_1),axis=0)
y_train = np.concatenate((y_airplane_1,y_ship_1,y_cat_1,y_horse_1,y_computer_1,y_desert_1,y_fish_1,y_flower_1,y_dog_1,y_fire_1),axis=0)

X_test  = np.concatenate((X_airplane_2,X_ship_2,X_cat_2,X_horse_2,X_computer_2,X_desert_2,X_fish_2,X_flower_2,X_dog_2,X_fire_2),axis=0)
y_test  = np.concatenate((y_airplane_2,y_ship_2,y_cat_2,y_horse_2,y_computer_2,y_desert_2,y_fish_2,y_flower_2,y_dog_2,y_fire_2),axis=0)

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

model = Sequential()
model.add(Input(shape=(len_v,)))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)

# WEBGRAFIA

#https://keras.io/guides/sequential_model/
#https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/#:~:text=You%20can%20save%20your%20NumPy,file%2C%20most%20commonly%20a%20comma.
