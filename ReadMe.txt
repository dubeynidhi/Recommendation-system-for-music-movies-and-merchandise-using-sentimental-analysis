In this project we tried to address the limitation of currently used recommendation systems that
usually rely on user’s search history and demographic information for making
recommendations. Though useful, we believe that in this era of rapidly growing user centric
digital developments, it becomes crucial to incorporate sentiments into the loop to make better
decisions and tailor certain experiences. Thus, we implemented a recommendation system that,
along with other factors, also considers user’s emotional state of mind into consideration. With
use, the system learns user’s preferences under different emotional states that helps cater to
their needs.

This recommendation system would incorporate sentiments derived from a text
which could be user’s post on a social media or user’s comment/reviews. There are three models: for music, movies and merchandise. The movies model
can read text, derive sentiments from the text such as happy, sad etc. and then can suggest
10 movies from the genre appropriate for that emotion. The model picks up a movie from the
pool of movies for the emotion suggested by text and then suggests 9 similar movies. If the
user picks up a movie that was not suggested by the model, then we add that movie in that pool
for that user making the model more user friendly. For the music recommendation model also,
we have implemented item and used based collaborative filtering. Item based suggests 5 similar
artists for an artist and user-based filtering suggests recommendation after looking at the user’s profile, mainly its age, country and previous music history. The merchandise recommendation model reads product reviews and then gives a score to the product which is used in
recommending the product to a new user.
