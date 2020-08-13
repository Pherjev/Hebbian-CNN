import numpy as np

K = 10
len_u = 1536 # 4096
W = np.zeros((K,len_u))
beta = 0.01

def S(W): # Activacion
        n,m = W.shape
        L = W >= 1
        L2= W < 1
        B = W*L
        O = np.ones((n,m))
        C = B + O*(W*L == 0.)
        D = np.log(C) + L*O
        D += W*L2
        return D

def logistic(W):
        return 1./(1.+np.exp(-W))

def HebbianTraining(X_train,y_train,W):
	#theta = np.ones((K,1))
	tau = 0.5
	theta = 1.
	for i in range(len(X_train)):
  		u = X_train[i] + np.random.rand(len_u)/10.
  		u = u.reshape((len_u,1))
		#u = u / np.max(u)
		#print(u)
  		v = np.zeros((K,1))
  		v[int(y_train[i]),0] = 1.

    		#W += np.matmul(v - logistic(np.matmul(S(W),u)),u.T) # PERCEPTRON
		#W += np.matmul(v,u.T) # BASIC HEBB
		theta += tau*(v**2 - theta) 
    		#W += np.matmul(v-1,u.T) # COVARIANCE
		#W += np.matmul(v,u.T)-beta*((v**2)*np.ones((K,len_u)))*S(W) # OJA
		W += np.matmul(v*(v-theta),u.T) # BCM 
		#v2 = logistic(np.matmul(S(W),u))
		#W += np.matmul(v2*(v2-theta),u.T)
  	return W

def HebbianTesting(X_test,y_test,W):
  	c = 0
  	for i in range(len(X_test)):
    		u = X_test[i] 
    		u = u.reshape((len_u,1))
		#u = u / np.max(u)
    		v = np.matmul(S(W),u)
		#v = np.matmul(W,u)
    		v = v.reshape(K)
    		v = np.argmax(v)
    		if v == y_test[i]:
      			c += 1
		#else:
		#	print(y_test[i],v)
		#	print(i)
 	return float(c)/len(X_test)

def resh(v):
	v1 = np.zeros((len(v),len_u))
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

#print(X_airplane_1.shape,X_airplane_2.shape)
#print(X_ship_1.shape,X_ship_2.shape)


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


X_train = np.concatenate((X_airplane_1,X_ship_1,X_cat_1,X_horse_1,X_computer_1,X_desert_1,X_fish_1,X_flower_1,X_dog_1,X_fire_1),axis=0)
y_train = np.concatenate((y_airplane_1,y_ship_1,y_cat_1,y_horse_1,y_computer_1,y_desert_1,y_fish_1,y_flower_1,y_dog_1,y_fire_1),axis=0)

X_test  = np.concatenate((X_airplane_2,X_ship_2,X_cat_2,X_horse_2,X_computer_2,X_desert_2,X_fish_2,X_flower_2,X_dog_2,X_fire_2),axis=0)
y_test  = np.concatenate((y_airplane_2,y_ship_2,y_cat_2,y_horse_2,y_computer_2,y_desert_2,y_fish_2,y_flower_2,y_dog_2,y_fire_2),axis=0)



#X_train = np.concatenate((X_cat_1,X_horse_1),axis=0)
#y_train = np.concatenate((y_cat_1,y_horse_1),axis=0)

#X_test  = np.concatenate((X_cat_2,X_horse_2),axis=0)
#y_test  = np.concatenate((y_cat_2,y_horse_2),axis=0)


W = HebbianTraining(X_train,y_train,W)
print("Training done...")
print(HebbianTesting(X_test, y_test,W))
