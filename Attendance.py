import cv2
import face_recognition

img1=face_recognition.load_image_file('images/obama.jpg')
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
testImg=face_recognition.load_image_file('images/obama2.jpg')
testImg=cv2.cvtColor(testImg,cv2.COLOR_BGR2RGB)
bidenImg=face_recognition.load_image_file('images/biden.jpg')
bidenImg=cv2.cvtColor(bidenImg,cv2.COLOR_BGR2RGB)



# -----------rectangle over img1------------ 

face=face_recognition.face_locations(img1)[0]
# print(face)              : returns tuple (142, 617, 409, 349)(y1,x2,x1,y2)

encodeFace=face_recognition.face_encodings(img1)[0]
# print(encodeFace)        :generatess 128 unique features with the help of dilib lib  

cv2.rectangle(img1,(face[3],face[0]),(face[1],face[2]),(0,255,0),3)




# -----------rectangle over img2 i.e testimg------------ 

testFace=face_recognition.face_locations(testImg)[0]
encodeTestFace=face_recognition.face_encodings(testImg)[0]
cv2.rectangle(testImg,(testFace[3],testFace[0]),(testFace[1],testFace[2]),(0,255,0),3)




# ---------------rectangle over biden img ---------------- 

bidenFace=face_recognition.face_locations(bidenImg)[0]
encodebidenFace=face_recognition.face_encodings(bidenImg)[0]
cv2.rectangle(bidenImg,(bidenFace[3],bidenFace[0]),(bidenFace[1],bidenFace[2]),(0,255,0),3)




# --------------compare faces-------------------- 
# comparing 2 obama images 
res=face_recognition.compare_faces([encodeFace],encodeTestFace)
face_distance=face_recognition.face_distance([encodeFace],encodeTestFace)
print(res,face_distance)

cv2.putText(testImg,f"{res} {round(face_distance[0],2)}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


# comparing obama and biden images 
res2=face_recognition.compare_faces([encodeFace],encodebidenFace)
face_distance=face_recognition.face_distance([encodeFace],encodebidenFace)
print(res2,face_distance)

cv2.putText(bidenImg,f"{res2} {round(face_distance[0],2)}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)







cv2.imshow("obama",img1)
cv2.imshow("obama test",testImg)
cv2.imshow("biden test",bidenImg)
cv2.waitKey()
cv2.destroyAllWindows()