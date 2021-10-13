# install the sparklyr package from CRAN as follows:

install.packages("dplyr")
install.packages("sparklyr")

# This is the R interface for Apache Spark - latest in distributed system technology.
# https://spark.rstudio.com/
# Purpose is quickly accessing and processing big data

library(sparklyr)
spark_install(version = "2.1.0")

# We can now call our good old dplyr for any data manipulation we might want to do.We can now use all of the available dplyr verbs against the tables within the cluster.

# You should also install a local version of Spark for development purposes:

sc <- spark_connect(master = "local")

# We can now use all of the available dplyr verbs against the tables within the cluster. We can copy some datasets from R into the Spark cluster 
library(dplyr)
install.packages(c("nycflights13", "Lahman"))



# copy three databases to sc clusters
iris_tbl <- copy_to(sc, iris)
flights_tbl <- copy_to(sc, nycflights13::flights, "flights")
batting_tbl <- copy_to(sc, Lahman::Batting, "batting")

#create a local spark cluster - check connection -  that we can now query
dplyr::src_tbls(sc)

# filter by departure delay and print the first few records
View(flights_tbl)
flights_tbl %>% filter(dep_delay == 2)

# Group by tail number - summarize the count, distance and delay - filter counts greater than 20 and distanc greater than 20000 and only those that are numbers

delay <- flights_tbl %>%
  group_by(tailnum) %>%
  summarise(count = n(), dist = mean(distance,na.rm=TRUE), delay = mean(arr_delay)) %>%
  filter(count > 20, dist < 2000, !is.na(delay)) %>%
  collect

# plot delays
library(ggplot2)
ggplot(delay, aes(dist, delay)) +
  geom_point(aes(size = count), alpha = 1/2) +
  geom_smooth() +
  scale_size_area(max_size = 2)

# Using SQL
# You can execute SQL queries directly against tables within a Spark cluster. The spark_connection object implements a DBI interface for Spark, so you can use dbGetQuery to execute SQL and return the result as an R data frame:

library(DBI)
iris_preview <- dbGetQuery(sc, "SELECT * FROM iris LIMIT 10")
iris_preview

# Can run many algorithms on the cluster data
# e.g  K-Means Clustering, Linear Regression, Logistic Regression,
# Generalized Linear Regression, Decision Trees, Random Forests, Gradient Boosted Trees, Principal Components Analysis etc


# Use Spark’s linear regression to model the linear relationship between a response variable and one or more explanatory variables.

lm_model <- iris_tbl %>%
select(Petal_Width, Petal_Length) %>%
  ml_linear_regression(Petal_Length ~ Petal_Width)

# Plotting predictions

iris_tbl %>%
  select(Petal_Width, Petal_Length) %>%
  collect %>%
  ggplot(aes(Petal_Length, Petal_Width)) +
  geom_point(aes(Petal_Width, Petal_Length), size = 2, alpha = 0.5) +
  geom_abline(aes(slope = coef(lm_model)[["Petal_Width"]],
                  intercept = coef(lm_model)[["(Intercept)"]]),
              color = "red") +
  labs(
    x = "Petal Width",
    y = "Petal Length",
    title = "Linear Regression: Petal Length ~ Petal Width",
    subtitle = "Use Spark.ML linear regression to predict petal length as a function of petal width."
  )


# Here we will demonstrate the use of Spark’s machine learning algorithms within R. We’ll use ml_linear_regression to fit a linear regression model. Using the built-in mtcars dataset, we’ll try to predict a car’s fuel consumption (mpg) based on its weight (wt), and the number of cylinders the engine contains (cyl).

mtcars_tbl <- copy_to(sc, mtcars, "mtcars", overwrite = TRUE)

# Transform the data with Spark SQL, feature transformers, and DataFrame functions.

# Use Spark SQL to remove all cars with horsepower less than 100
# Use Spark feature transformers to bucket cars into two groups based on cylinders
# Use Spark DataFrame functions to partition the data into test and training
# Then fit a linear model using spark ML. Model MPG as a function of weight and cylinders.

# transform our data set, and then partition into 'training', 'test'
partitions <- mtcars_tbl %>%
  filter(hp >= 100) %>%
  mutate(cyl8 = cyl == 8) %>%
  sdf_random_split(training = 0.5, test = 0.5, seed = 888)

# fit a linear model to the training dataset
fit <- partitions$training %>%
  ml_linear_regression(mpg ~ wt + cyl)
# summarize the model
summary(fit)
# Score the data
pred <- ml_predict(fit, partitions$test) %>%
  collect

glimpse(pred)
# Plot the predicted versus actual mpg

ggplot(pred, aes(x = mpg, y = prediction)) +
  geom_abline(lty = "dashed", col = "red") +
  geom_point() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_fixed(ratio = 1) +
  labs(
    x = "Actual Fuel Consumption",
    y = "Predicted Fuel Consumption",
    title = "Predicted vs. Actual Fuel Consumption"
  )

# Although simple, our model appears to do a fairly good job of predicting a car’s average fuel consumption.
# As you can see, we can easily and effectively combine feature transformers, machine learning algorithms, and Spark DataFrame functions into a complete analysis with Spark and R.

################################################################################
### Once we are done with our analyses we will close the spark connection
spark_disconnect(sc)

###############################################################################
#### REMOTE CONNECTION USING LIVY ##############################################
#Livy enables remote connections to Apache Spark clusters. Before connecting to Livy,
#you will need the connection information to an existing service running Livy.
#Otherwise, to test livy in your local environment, you can install it and run it locally as follows:

livy_install()
livy_service_start()

#To connect, use the Livy service address as master and method = "livy" in spark_connect
sc <- spark_connect(master = "http://localhost:8998", method = "livy")
copy_to(sc, iris)

#To stop livy connection
spark_disconnect(sc)
livy_service_stop()

#To connect to remote livy clusters, use authentications as below
config <- livy_config(username="username", password="password")
sc <- spark_connect(master = "<address>", method = "livy", config = config)
spark_disconnect(sc)