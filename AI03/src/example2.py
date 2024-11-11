from faceDetect import faceDetect

def example2():

    # faceDetect 객체 생성시 인자로 사진들을 저장할 데이터베이스 경로를 입력합니다.
    fd=faceDetect("./image")

    # fd.current_face()함수는 현재 얼굴 데이터를 반환해주는 함수입니다.
    cur_img=fd.current_face()

    # 데이터베이스(./image)에 있는 얼굴들과 cur_img를 비교합니다.
    result = fd.match_face(cur_img)
    print(result)

    result = fd.match_face(cur_img)





if __name__ == '__main__':
    example2()