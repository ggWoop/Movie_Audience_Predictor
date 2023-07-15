import numpy as np
import pandas as pd
import streamlit as st
import csv
import openai
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from supabase import create_client

openai.api_key = st.secrets.OPENAI_TOKEN
openai_model_version = "gpt-3.5-turbo-0613"

# Supabase setup
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase_client = init_connection()



# # CSS
# st.markdown("""
#     <style>
#         div[data-baseweb="input"] > div {
#             background-color: #212425 !important;
#         }
#         div[data-baseweb="select"] > div {
#             background-color: #212425 !important;
#         }
#         div[role="listbox"] ul {
#             background-color: #212425 !important;
#         }
#     </style>
#     """, unsafe_allow_html=True)





with open('./css/wave.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# 저장된 피클 파일 경로
pickle_file = './data/dictionary_data.pkl'


# 피클 파일로부터 딕셔너리 데이터 로드
with open(pickle_file, 'rb') as file:
    dictionary_data = pickle.load(file)



def read_unique_nationalities(file_path):
    nationalities = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)

        for row in reader:
            nationality = row[0]  # 국적은 첫 번째 열에 있을 것으로 가정합니다.
            nationalities.append(nationality)
    return nationalities




def generate_prompt(title, genres, director, cast, screenplay, original_work,
                    runtime, rating, num_screens, nationality,
                    series_value,corona_value,total_data):
    prompt = f"""
            제목: {title}
            장르: {''.join(genres)}
            감독: {director}
            주연: {''.join(cast)}
            각본: {screenplay}
            원작: {original_work}
            런타임: {runtime}
            등급: {rating}
            스크린수: {num_screens}
            시리즈의 여부: {series_value}
            코로나의 이후의 영화:{corona_value}
            영화의 국적: {nationality}
            머신러닝 으로 예측한 관객수: {total_data}
    ------------------------------------------
    저희는 머신러닝을 이용해 사용자로부터 위 데이터를 입력받아 아직 상영되지 않은 영화의 관객수를 예측합니다.


    위 데이터를 바탕으로 왜 이 영화가 어떻게 저 관객수가 나왔는지 자세히 설명해주세요.
    당신의 생각을 자유롭게 말해도 좋습니다.
    위에 써져있는 데이터들을 말해주지 말고 최대한 설명 해주세요.

    ------------------------------------------
    시리즈의 여부: 가 1이면 시리즈가 있다고 생각해주세요.
    코로나의 이후의 영화: 가 1이면 코로나 이후의 영화입니다.
    """


    return prompt.strip()






def request_chat_completion(prompt):
    response = openai.ChatCompletion.create(
        model=openai_model_version,
        messages=[
            {"role": "system", "content": "당신은 전문 영화 상영 전문가 입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]


def get_avg_audience_by_person_and_date(person, date, role):
    if role == 'actor':
        person_dict = actor_avg_audience.get(person, {})
    elif role == 'director':
        person_dict = director_avg_audience.get(person, {})
    elif role == 'scriptwriter':
        person_dict = scriptwriter_avg_audience.get(person, {})
    elif role == 'writer':
        person_dict = writer_avg_audience.get(person, {})
    else:
        return 0

    closest_date = None
    for release_date in person_dict.keys():
        if release_date <= pd.Timestamp(date) and (closest_date is None or release_date > closest_date):
            closest_date = release_date

    if closest_date is not None:
        return person_dict[closest_date]
    else:
        return 0



def get_highest_avg_audience(date, role, limit=50):
    # 선택한 역할에 따라 해당하는 딕셔너리 선택
    if role == 'actor':
        data_dict = actor_avg_audience
    elif role == 'director':
        data_dict = director_avg_audience
    elif role == 'scriptwriter':
        data_dict = scriptwriter_avg_audience
    elif role == 'writer':
        data_dict = writer_avg_audience
    else:
        return []

    # 선택한 날짜 이전까지의 평균 관객 수 가져오기
    avg_audience_dict = {}
    for person, data in data_dict.items():
        max_date = pd.to_datetime('1900-01-01')
        avg_audience = 0.0
        for date_key, audience in data.items():
            curr_date = pd.to_datetime(date_key)
            if curr_date <= date and curr_date > max_date:
                max_date = curr_date
                avg_audience = float(audience) if audience != '' else 0.0
        avg_audience_dict[person] = avg_audience

    # 평균 관객 수를 기준으로 내림차순 정렬
    avg_audience = [(person, avg_audience) for person, avg_audience in avg_audience_dict.items()]
    avg_audience.sort(key=lambda x: x[1], reverse=True)

    # 상위 limit개의 사람 이름과 관객 수의 리스트 반환
    return avg_audience[:limit]



def genre_to_onehot(input_genres):
    # 장르 컬럼 이름 목록
    genre_columns = ['genre_ 스릴러', 'genre_ 액션', 'genre_SF', 'genre_가족', 'genre_공연', 'genre_공포',
                     'genre_기타', 'genre_다큐멘터리', 'genre_드라마', 'genre_로맨스', 'genre_뮤지컬',
                     'genre_미스터리', 'genre_범죄', 'genre_사극', 'genre_서부극', 'genre_성인물',
                     'genre_스릴러', 'genre_애니메이션', 'genre_액션', 'genre_어드벤처', 'genre_전쟁',
                     'genre_코미디', 'genre_판타지']

    # 모든 장르에 대해 0으로 초기화된 딕셔너리 생성
    genre_dict = {genre: [0] for genre in genre_columns}

    # 입력받은 장르에 대해 값 설정
    for genre in input_genres:  # split() 메서드를 사용하지 않고 리스트를 직접 순회
        genre = 'genre_' + genre.strip()  # 앞뒤 공백 제거 및 prefix 추가
        if genre in genre_dict:
            genre_dict[genre] = [1]  # 해당 장르 값 설정

    return genre_dict





genres_list = ['스릴러', '액션', 'SF', '가족', '공연', '공포', '기타', '다큐멘터리', '드라마', '로맨스', '뮤지컬', '미스터리', '범죄', '사극', '서부극',
                   '성인물', '애니메이션', '어드벤처', '전쟁', '코미디', '판타지']

nationality_list = ['기타', '미국_캐나다', '유럽', '일본', '중국_대만_홍콩', '한국']

rating_list = ['전체관람가', '12세관람가', '15세관람가', '청소년관람불가']




# 각 딕셔너리에 접근
actor_avg_audience = dictionary_data['actor_avg_audience']
director_avg_audience = dictionary_data['director_avg_audience']
scriptwriter_avg_audience = dictionary_data['scriptwriter_avg_audience']
writer_avg_audience = dictionary_data['writer_avg_audience']

st.sidebar.markdown(
    """
<iframe width="300" height="170" src="https://www.youtube.com/embed/jmF0edcnw4w?autoplay=1" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
""",
    unsafe_allow_html=True,
)

from PIL import Image

image = Image.open('./data/cinema.jpeg')
st.image(image, use_column_width=True)

st.markdown("## CineInsight : 당신의 영화를 에측 해드립니다. 🎥")
st.markdown('***')



tab1, tab2 = st.tabs(["영화 관객수 에측", "영화인 평균 관객수 보기"])

with tab1:
    # 제목 입력
    title = st.text_input("영화 제목을 입력하세요.")

    # 장르 선택

    genres = st.multiselect("장르를 선택하세요. (최대 3개)", genres_list, max_selections=3)

    # # 감독 선택
    # director = st.selectbox("감독을 선택하세요.", list(director_list), index=director_list.index('봉준호'))
    #
    # # 주연 선택 (최대 3명까지)
    # cast = st.multiselect("주연 배우를 선택하세요. (최대 3명)", list(actor_list), max_selections=3)

    # 감독 선택
    director_list = list(dictionary_data['director_avg_audience'].keys())
    director = st.selectbox("감독을 선택하세요.", director_list)

    # 주연 선택 (최대 3명까지)
    actor_list = list(dictionary_data['actor_avg_audience'].keys())
    cast = st.multiselect("주연 배우를 선택하세요. (최대 3명)", actor_list, max_selections=3)




    col1, col2 = st.columns(2)

    with col1:
        # screenplay = st.text_input("각본 작가의 이름을 입력하세요.")
        # 각본 작가 선택
        scriptwriter_list = list(dictionary_data['scriptwriter_avg_audience'].keys())
        screenplay = st.selectbox("각본 작가를 선택하세요.", scriptwriter_list)

    with col2:
        # original_work = st.text_input("원작자의 이름을 입력하세요.")
        # 작가 선택
        writer_list = list(dictionary_data['writer_avg_audience'].keys())
        # "주호민"의 index 찾기
        default_index = writer_list.index("주호민") if "주호민" in writer_list else 0

        original_work = st.selectbox("원작자를 선택하세요.", writer_list, index=default_index)

    col1, col2 = st.columns(2)

    with col1:
        # 런타임 입력
        runtime = st.number_input("상영 시간을 입력하세요. (분 단위)", min_value=1)

        # 입력한 시간을 카테고리로 변환
        if runtime < 90:
            runtime_category = 1
        elif runtime <= 110:
            runtime_category = 2
        else:
            runtime_category = 3

    with col2:
        # 스크린수 입력
        num_screens = st.number_input("상영 스크린 수를 입력하세요.", min_value=1)

    col1, col2 = st.columns(2)

    with col1:
        # 등급 선택

        rating = st.selectbox("영화 등급을 선택하세요.", rating_list)

    with col2:
        # 국적 선택

        nationality = st.selectbox("영화의 국적을 선택하세요.", nationality_list, index=5)


    col1, col2 = st.columns(2)

    with col1:
        # 시리즈물인지 입력받음
        is_series = st.checkbox("이 영화는 시리즈물입니까?")

        # 체크박스의 값을 이용하여 0 또는 1로 저장
        series_value = 1 if is_series else 0

    with col2:

        # 시리즈물인지 입력받음
        is_corona = st.checkbox("이 영화는 코로나 이후에 개봉 했습니까? (2019년 11월 17일)")

        # 체크박스의 값을 이용하여 0 또는 1로 저장
        corona_value = 1 if is_corona else 0
    st.markdown('***')

    # 입력한 데이터 출력
    if st.button("영화 관객수 예측"):
        if not genres:
            st.warning("적어도 하나 이상의 장르를 선택해야 합니다.")
            st.stop()
        if not cast:
            st.warning("적어도 하나 이상의 주연 배우를 선택해야 합니다.")
            st.stop()

        genres_str = ", ".join(genres)
        cast_data = ", ".join(cast)


        # cast_list 생성
        cast_list = [name.strip() for name in cast]  # 공백 제거


        total_actor_avg_audience = []
        for name in cast_list:
            avg_audience = get_avg_audience_by_person_and_date(name, '2023-07-10', 'actor')
            total_actor_avg_audience.append(avg_audience)  # 리스트로 저장

        director_avg_audience = get_avg_audience_by_person_and_date(director, '2023-07-10', 'director')
        scriptwriter_avg_audience = get_avg_audience_by_person_and_date(director, '2023-07-10', 'scriptwriter')
        writer_avg_audience = get_avg_audience_by_person_and_date(director, '2023-07-10', 'writer')



        with open('./data/movie_scaler.pkl', 'rb') as file:
            scaler_dict = pickle.load(file)

        # actor_changed 컬럼만 먼저 스케일링
        scaled_actor_avg_audience = scaler_dict['actor_changed'].transform(
            np.array(total_actor_avg_audience).reshape(-1, 1))
        scaled_actor_changed = sum(scaled_actor_avg_audience)
        # 새로운 데이터 생성
        new_data = pd.DataFrame({
            'actor_changed': [scaled_actor_changed],  # 스케일링된 값의 합 사용
            'screens': [num_screens],
            'director_changed': [director_avg_audience],
            'scriptwriter_changed': [scriptwriter_avg_audience],
            'writer_changed': [writer_avg_audience],
        })

        # 나머지 컬럼 스케일링
        for column in ['screens', 'director_changed', 'scriptwriter_changed', 'writer_changed']:
            new_data[column] = scaler_dict[column].transform(new_data[[column]])

        # st.write(new_data)
        # st.write(genre_to_onehot(genres))

        genre_data = pd.DataFrame(genre_to_onehot(genres))

        # 두 데이터프레임 합치기
        new_data = pd.concat([new_data, genre_data], axis=1)
        total_data = pd.DataFrame(genre_data)



        # 딕셔너리 생성
        extra_data = {
            'running_time': [runtime],
            'series_0': [1 if series_value == 0 else 0],
            'series_1': [1 if series_value == 1 else 0],
            'corona_0': [1 if corona_value == 0 else 0],
            'corona_1': [1 if corona_value == 1 else 0],
            '기타': [1 if nationality == '기타' else 0],
            '미국_캐나다': [1 if nationality == '미국_캐나다' else 0],
            '유럽': [1 if nationality == '유럽' else 0],
            '일본': [1 if nationality == '일본' else 0],
            '중국_대만_홍콩': [1 if nationality == '중국_대만_홍콩' else 0],
            '한국': [1 if nationality == '한국' else 0],
            '12세관람가': [1 if rating == '12세관람가' else 0],
            '15세관람가': [1 if rating == '15세관람가' else 0],
            '전체관람가': [1 if rating == '전체관람가' else 0],
            '청소년관람불가': [1 if rating == '청소년관람불가' else 0]
        }

        # new_data에 추가
        for key, value in extra_data.items():
            new_data[key] = value

        # DataFrame으로 변환
        total_data = pd.DataFrame(new_data)
        # 원래 데이터의 피처 순서에 맞게 재정렬
        ordered_columns = ['running_time', 'screens', 'actor_changed', 'director_changed', 'scriptwriter_changed',
                           'writer_changed', 'genre_ 스릴러', 'genre_ 액션', 'genre_SF', 'genre_가족', 'genre_공연',
                           'genre_공포',
                           'genre_기타', 'genre_다큐멘터리', 'genre_드라마', 'genre_로맨스', 'genre_뮤지컬', 'genre_미스터리',
                           'genre_범죄',
                           'genre_사극', 'genre_서부극', 'genre_성인물', 'genre_스릴러', 'genre_애니메이션', 'genre_액션',
                           'genre_어드벤처',
                           'genre_전쟁', 'genre_코미디', 'genre_판타지', 'series_0', 'series_1', '기타', '미국_캐나다', '유럽', '일본',
                           '중국_대만_홍콩', '한국', '12세관람가', '15세관람가', '전체관람가', '청소년관람불가', 'corona_0', 'corona_1']
        total_data = total_data.reindex(columns=ordered_columns)

        with open('./data/model.pkl', 'rb') as file:
            model = pickle.load(file)

        # 예측
        predicted = model.predict(total_data)



        formatted_predicted = "{:,}".format(int(predicted[0]))

        st.markdown(f"## 당신의 영화의 예상 관객수는 : {formatted_predicted} 명 입니다.")


        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        df = pd.read_csv('./data/preprocessed.csv')

        # 모델이 반환한 예상 관객수
        predicted_audience = float(formatted_predicted.replace(',', ''))



        fig, ax = plt.subplots()

        sns.kdeplot(data=df, x="audience", bw_adjust=1.5, color='b', ax=ax, fill=True)
        ax.set_xlabel('관객수')
        ax.set_ylabel('확률밀도')
        ax.set_xscale('log')

        log_audience = np.log10(df['audience'])

        log_predicted_audience = np.log10(predicted_audience)

        closest_index = (np.abs(log_audience - log_predicted_audience)).idxmin()
        percentile = (closest_index / (len(df) - 1)) * 100

        plt.axvline(predicted_audience, color='red', linestyle='--',
                    label=f"예상 관객수: {predicted_audience:,.0f}명\n  (전체 상위 {percentile:.2f}%)")
        legend = ax.legend(bbox_to_anchor=(0.95, 0.7), loc='upper right', ncol=1)
        legend.get_texts()[0].set_color('green')


        def human_readable_number(x, pos):
            if x >= 1e6:
                return f"{x * 1e-6:.0f}M"
            elif x >= 1e3:
                return f"{x * 1e-3:.0f}k"
            else:
                return f"{x:.0f}"


        ax.xaxis.set_major_formatter(ticker.FuncFormatter(human_readable_number))
        ax.tick_params(axis='x', labelrotation=45)

        plt.title('영화 관객수 분포')

        # Streamlit에 plot을 표시
        st.pyplot(fig)

        st.markdown('***')

        with st.spinner('AI가 예측중입니다...'):
            prompt = generate_prompt(title, genres, director, ', '.join(cast), screenplay, original_work, runtime, rating,
                                     num_screens, nationality,series_value,corona_value,total_data)
            ai_response = request_chat_completion(prompt)

            # st.text_area(
            #     label="영화 관객수 예측 결과",
            #     value=ai_response,
            #     placeholder="아직 에측되지 않았습니다.",
            #     height=600
            # )
            #

            st.markdown("<h2 style='text-align: left;'>영화 관객수 예측 결과</h2>", unsafe_allow_html=True)
            st.text_area(
                label="",
                value=ai_response,
                placeholder="아직 예측되지 않았습니다.",
                height=600
            )

        movie_data = {
            "title": title,
            "genres": genres_str,
            "director": director,
            "cast": cast_data,
            "screenplay": screenplay,
            "original_work": original_work,
            "runtime": runtime,
            "rating": rating,
            "num_screens": num_screens,
            "nationality": nationality,
            "series_value": series_value,
            'corona_value' : corona_value,
            "gpt": ai_response,

        }


        response = supabase_client.table("movie_audience_forecast").insert(movie_data).execute()



with tab2:



    st.markdown('<span style="color:white;font-size:35px;">특정인 특정시점 평균관객수 출력</span>', unsafe_allow_html=True)

    with st.container():
        choice = st.selectbox("직업을 선택하세요.", ['actor', 'director', 'scriptwriter', 'writer'], key='choice1')

        # 선택한 직업에 따라 데이터 선택
        if choice == 'actor':
            selected_data = dictionary_data['actor_avg_audience']
        elif choice == 'director':
            selected_data = dictionary_data['director_avg_audience']
        elif choice == 'scriptwriter':
            selected_data = dictionary_data['scriptwriter_avg_audience']
        elif choice == 'writer':
            selected_data = dictionary_data['writer_avg_audience']

        # 선택한 데이터로부터 감독 목록 생성
        director_list = list(selected_data.keys())
        director_p2 = st.selectbox("감독을 선택하세요.", director_list, key='director_p2')

        selected_date = st.date_input("날짜를 선택하세요.")


        avg_audience = get_avg_audience_by_person_and_date(director_p2, selected_date, choice)
        st.write(f"선택한 날짜에 {director_p2}의 평균 관객 수는 {avg_audience:,}명 입니다.")


    st.markdown("---")

    with st.container():
        st.markdown('<span style="color:white;font-size:35px;">특정시점 특정역할 상위명단 출력</span>', unsafe_allow_html=True)

        
        # 특정 날짜와 역할 입력
        selected_date_p3 = st.date_input("날짜를 선택하세요.",key='selected_date_p3')

        choice_p3 = st.selectbox("직업을 선택하세요.", ['actor', 'director', 'scriptwriter', 'writer'], key='choice2')

        # 선택한 직업에 따라 데이터 선택
        if choice_p3 == 'actor':
            selected_data = dictionary_data['actor_avg_audience']
        elif choice_p3 == 'director':
            selected_data = dictionary_data['director_avg_audience']
        elif choice_p3 == 'scriptwriter':
            selected_data = dictionary_data['scriptwriter_avg_audience']
        elif choice_p3 == 'writer':
            selected_data = dictionary_data['writer_avg_audience']

        # 특정 날짜와 역할에 대한 평균 관객 수가 가장 높은 순으로 상위 50개 사람 이름과 관객 수 리스트 출력

        result = get_highest_avg_audience(selected_date_p3, choice_p3, limit=10)

        for idx, (person, avg_audience) in enumerate(result, start=1):
            formatted_avg_audience = "{:,.0f}".format(avg_audience)
            st.write(f" {idx}. {person}")









