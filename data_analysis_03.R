#### Ideas / Notes  -------------------------------------------------

'The rapid growth and fierce competition among content creators on platforms like YouTube
necessitate a deeper understanding of the factors that drive viewer engagement. With a
staggering 79% of views concentrated among just 10% of videos, as highlighted by a Pew
Research Centre analysis, the pressure to create captivating content is immense. This thesis
addresses this challenge by exploring the quantitative impact of video content elements,
specifically focusing on titles and thumbnails, on engagement metrics such as views, likes, and
comments.
Building on the theoretical insights of Jonah Berger and colleagues, which emphasize the
importance of emotional dynamics in content engagement, this research integrates both
statistical and machine learning approaches to provide a comprehensive analysis. The data
pipeline established in this study not only gathers YouTube video information via the official
API but also enriches it with additional quantitative aspects, including video title sentiment,
length, and thumbnail features such as the presence of faces, logos, and text, as well as facial
features like gender and emotion.
Exploratory data analysis, followed by correlation analysis and ANOVA, laid the groundwork
for understanding the relationships between these factors and video performance metrics.
The linear regression models and random forest machine learning techniques used in this
thesis revealed significant predictors of engagement. For instance, variables such as video
duration, semantic similarity between video title and thumbnail as well as colourfulness of
the thumbnail emerged as top predictors. The findings were further validated by comparing
them with existing research, underscoring the alignment and extending our understanding of
video content optimization.
This thesis not only demonstrates the predictive power of these models but also highlights
areas for further refinement and exploration. Future research could benefit from integrating
specialized models trained specifically on thumbnail data and exploring new statistical
modelling approaches to improve predictive accuracy. By understanding and leveraging the
key factors that influence video performance, content creators and marketers can optimize
their strategies to enhance engagement and achieve better outcomes on the platform.
The findings offer practical recommendations for content creators and marketers,
emphasizing the importance of emotional dynamics and strategic content design in driving
viewer engagement. This research contributes to the broader field of cultural analytics,
offering new perspectives on what makes content truly engaging.'

#### Load Libraries -------------------------------------------------
library(dplyr)
library(stringr)
library(e1071)
library(ggplot2)
library(GGally)
library(grid)
library(gridExtra)
library(MuMIn)
library(lmtest)
library(mgcv)
library(car)
library(MASS)
library(rpart)
library(rpart.plot)
library(caret)
library(sandwich)
library(randomForest)
library(multcomp)
library(multcompView)
library(ranger)
library(pdp)
library(rlang)
library(tidyr)
library(broom)
library(purrr)

#Option in the sense not mentioned in the thesis (not essential for the final research)

library(vader)
library(tidyverse)
library(readxl)
library(nnet)
library(multinom)
library(boot)
library(DAAG)


#### Check and set Working Directory -------------------------------------------
setwd("/Users/folder") #ADJUST IF NEEDED

#### Read Data -----------------------------------------------------------------
data_raw <- read.csv("/Users/video_platform_engagement_data.csv") #ADJUST IF NEEDED
dim(data_raw) 

#### 00 | DATA CLEAN ------------------------------------------------------------
data_raw = na.omit(data_raw) # Remove Na's
dim(data_raw)


data_raw <- data_raw %>%
  dplyr::filter(Video.Views.Normalized != 0 & Video.Likes != 0 & Video.Comments != 0 ) #Removing YT video statistic which are private or disabled or should not be possible
dim(data_raw)

#Find Category Name and ID pairs
video_categories <- aggregate(Category.Name ~ Category.ID, data = data_raw, FUN = function(x) unique(as.character(x)))

boxcox_result <- boxcox(lm(data_raw$Video.Views ~ 1)) # [OPTIONAL] Use boxcox to find the lambda value to transform views accordingly
lambda <- boxcox_result$x[which.max(boxcox_result$y)]

data_clean <- data_raw %>%
  dplyr::mutate(
    TN.Human.Faces.Emotion = str_to_lower(if_else(All.Emotion == "", "none", All.Emotion)),
    TN.Human.Faces.Gender = str_to_lower(if_else(All.Gender == "", "none", All.Gender)),
    TN.Human.Face.Count = Faces.Rekognition,
    Video.Views.Boxcox = round((data_raw$Video.Views ^ lambda - 1) / lambda,3),
    Video.Views.Grouped = case_when(
      Video.Views < 15000                           ~ "verylow",
      Video.Views >= 15000 & Video.Views <= 49999         ~ "low",
      Video.Views >= 50000 & Video.Views <= 249999        ~ "medium",
      Video.Views >= 250000 & Video.Views <= 999999       ~ "high",
      Video.Views >= 1000000 & Video.Views <= 9999999     ~ "veryhigh",
      Video.Views >= 10000000                       ~ "extreme"
    ),
    Video.Views.Normalized.Log = round(log(Video.Views.Normalized), 4),
    Video.Views.Log = round(log(Video.Views), 2),
    Video.Likes.Log = round(log(Video.Likes), 2),
    Video.Comments.Log = round(log(Video.Comments), 2),
    Video.Shallow.Engagement = round((Video.Likes/Video.Views),4),
    Video.Deep.Engagement = round((Video.Comments/Video.Views),4),
    Video.Shallow.Engagement.CubicRoot = round((Video.Likes/Video.Views)^(1/3),4),
    Video.Deep.Engagement.EighthRoot = round((Video.Comments/Video.Views)^(1/8),4),
    Subscriber.Count.Log = round(log(Subscriber.Count + 1), 2),
    Subscriber.Count = round(Subscriber.Count, 2),
    TN.Human.Faces = if_else(Faces.Rekognition == 0, "no", "yes"),
    TN.Logos = if_else(Logos.Detected == 0, "no", "yes"),
    TN.Logos.Count = Logos.Detected,
    TN.Text = if_else(Text.Detected == "yes", "yes", "no"),
    Video.Title.Emoji = if_else(Emoji.Count == 0, "no", "yes"),
    Video.Title.Emoji.Count = Emoji.Count,
    Video.Duration.Sec.Log = round(log(Duration..seconds. + 1), 2),
    Video.Duration.Sec = round(Duration..seconds., 2),
    Semantic.Similarity = round(Semantic.Similarity, 4),
    Colorfulness.Score.SquareRoot = round(Colorfulness.Score^(1/2), 3),
    Video.Title.Sentiment.Vader.Cluster = dplyr::case_when(
      Video.Title.Sentiment.Vader <= -0.15 ~ "negative",
      Video.Title.Sentiment.Vader >= 0.15 ~ "positive",
      TRUE ~ "neutral"
    )
  ) %>%
  dplyr::select(
    -Video.Title.Sentiment, -Emoji.Count, -Logos.Detected,
    -Duration..seconds., -Raw.API.Duration, -Detected.Logos, -Text.Detected,
    -Detected.Text, -All.Gender, -All.Emotion, -Faces.Rekognition,
    -Channel.Name, -Channel.ID, -Video.Description, -Category.Name,
    -Video.Likes, -Video.Comments, -Video.Likes.Normalized,
    -Video.Comments.Normalized, -Video.Title.Sentiment.Vader, -TN.Human.Face.Count, 
    -TN.Logos.Count, -Video.Title.Emoji.Count, -Video.Views.Boxcox, -Video.Views.Grouped, -Video.Views.Normalized.Log
  )

# Adjusting the factor levels for each variable
data_clean$TN.Human.Faces <- factor(data_clean$TN.Human.Faces, levels = c("no", "yes"))
data_clean$TN.Logos <- factor(data_clean$TN.Logos, levels = c("no", "yes"))
data_clean$TN.Text <- factor(data_clean$TN.Text, levels = c("no", "yes"))
data_clean$Video.Title.Emoji <- factor(data_clean$Video.Title.Emoji, levels = c("no", "yes"))
data_clean$Video.Title.Sentiment.Vader.Cluster <- factor(data_clean$Video.Title.Sentiment.Vader.Cluster, levels = c("negative", "neutral","positive"))
data_clean$Category.ID <- factor(data_clean$Category.ID, levels = video_categories$Category.ID, labels = video_categories$Category.Name)
data_clean$TN.Human.Faces.Emotion <- factor(data_clean$TN.Human.Faces.Emotion, levels = c("none", "calm", "happy", "sad", "angry", "surprised", "disgusted", "fear", "confused", "mixed"))
data_clean$TN.Human.Faces.Gender <- factor(data_clean$TN.Human.Faces.Gender, levels = c("none", "male", "female", "mixed"))

dim(data_clean)
summary(data_clean)

# DISTRIBUTION and OUTLIER ANALYSIS
calculate_full_boxplot_stats <- function(data, columns) {
  # Initialize an empty data frame to store results
  results_df <- data.frame(
    Variable = character(),
    Min = numeric(),
    Q1 = numeric(),
    Median = numeric(),
    Q3 = numeric(),
    Max = numeric(),
    NumOutliers = integer(),
    LowerCI = numeric(),
    UpperCI = numeric(),
    Skewness = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Loop through each variable in the columns list
  for (var in columns) {
    # Calculate the boxplot statistics, but do not plot it
    bp_stats <- boxplot(data[[var]], plot = FALSE)
    
    # Append the results to the results DataFrame
    results_df <- rbind(results_df, data.frame(
      Variable = var,
      Min = bp_stats$stats[1],
      Q1 = bp_stats$stats[2],
      Median = bp_stats$stats[3],
      Q3 = bp_stats$stats[4],
      Max = bp_stats$stats[5],
      NumOutliers = length(bp_stats$out),
      LowerCI = bp_stats$conf[1],
      UpperCI = bp_stats$conf[2],
      Skewness = skewness(data[[var]], type = 3)  # Skewness calculation
    ))
  }
  
  # Return the results data frame
  return(results_df)
}

continuous_columns <- c("Video.Views.Log", "Video.Shallow.Engagement.CubicRoot", "Video.Deep.Engagement.EighthRoot", 
                        "Subscriber.Count.Log", "Colorfulness.Score.SquareRoot", "Semantic.Similarity", 
                        "Video.Title.Length", "Video.Duration.Sec.Log")

results <- calculate_full_boxplot_stats(data_clean, continuous_columns) 
print(results)

# Plot all Boxplot
par(mfrow=c(2,4)) 
boxplot(data_clean$Video.Views.Log, main = "Video Views Log")
boxplot(data_clean$Video.Shallow.Engagement.CubicRoot, main = "Video Shallow Engagement Cubic Root")
boxplot(data_clean$Video.Deep.Engagement.EighthRoot, main = "Video Deep Engagement Eighth Root")
boxplot(data_clean$Subscriber.Count.Log, main = "Subscriber Count Log")
boxplot(data_clean$Colorfulness.Score.SquareRoot, main = "Colorfulness Score Square Root")
boxplot(data_clean$Semantic.Similarity, main = "Semantic Similarity")
boxplot(data_clean$Video.Title.Length, main = "Video Title Length")
boxplot(data_clean$Video.Duration.Sec.Log, main = "Video Duration Sec Log")




# REMOVE OUTLIERS
identify_outliers <- function(variable) {
  q1 <- quantile(variable, 0.25, na.rm = TRUE)
  q3 <- quantile(variable, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  return(variable < lower_bound | variable > upper_bound)
}

outlier_flags <- sapply(data_clean[continuous_columns], identify_outliers)
outlier_rows <- apply(outlier_flags, 1, any) # Reduce to a single logical vector where TRUE indicates an outlier in any column
cat("Rows containing outliers in any variable: ", sum(outlier_rows), "\n")

# Filter out
data_clean <- data_clean[!outlier_rows, ]
dim(data_clean)


#### 01 | DATA ANALYSIS/VISUALIZATION -------------------

# PLOT ALL INDEPENDENT VARIABLES
# Group 1
vid_cat <- ggplot(data_clean, aes(x = reorder(Category.ID, Category.ID, function(x) -length(x)))) + geom_bar(fill = "steelblue", color = "black") + labs(title = "Video Categories", x = "Category", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
vid_cat

# Group 2
vid_face <- ggplot(data_clean, aes(x = TN.Human.Faces)) + geom_bar(fill = "steelblue", color = "black") + labs(title = "Presence of Human Faces in Thumbnails", x = "Human Face Detected", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
vid_logo <- ggplot(data_clean, aes(x = TN.Logos)) + geom_bar(fill = "steelblue", color = "black") + labs(title = "Presence of Logos in Thumbnails", x = "Logo(s) Detected", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
vid_text <- ggplot(data_clean, aes(x = TN.Text)) + geom_bar(fill = "steelblue", color = "black") + labs(title = "Presence of Text in Thumbnails", x = "Text Detected", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
vid_title_emoji <- ggplot(data_clean, aes(x = Video.Title.Emoji)) + geom_bar(fill = "steelblue", color = "black") + labs(title = "Presence of Emojis in Video Title", x = "Emoji Detected", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
grid.arrange(vid_face, vid_logo, vid_text, vid_title_emoji, ncol=2, nrow=2) # All Yes/No Variables

# Group 3 
vid_face_g <- ggplot(data_clean, aes(x = reorder(TN.Human.Faces.Gender, TN.Human.Faces.Gender, function(x) -length(x)))) + geom_bar(fill = "steelblue", color = "black") + labs(title = "Gender on Thumbnail", x = "Gender", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 14, hjust = 0.5)) 
vid_face_e <- ggplot(data_clean, aes(x = reorder(TN.Human.Faces.Emotion, TN.Human.Faces.Emotion, function(x) -length(x)))) + geom_bar(fill = "steelblue", color = "black") + labs(title = "Emotions on Thumbnail", x = "Emotion", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 14, hjust = 0.5)) 
vid_sent_vader_clust <- ggplot(data_clean, aes(x = reorder(Video.Title.Sentiment.Vader.Cluster, Video.Title.Sentiment.Vader.Cluster, function(x) -length(x)))) + geom_bar(fill = "steelblue", color = "black") + labs(title = "Video Title Sentiment", x = "Sentiment Group", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 14, hjust = 0.5)) 
grid.arrange(vid_face_g, vid_face_e, vid_sent_vader_clust, ncol=3, nrow=1) # Some more for Face

# Group 4
vid_color <- ggplot(data_clean, aes(x = Colorfulness.Score.SquareRoot)) + geom_histogram(bins = 30, fill = "steelblue", color = "black") + labs(title = "Colorfulness", x = "Root of Colorfulness Score", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
vid_sim <- ggplot(data_clean, aes(x = Semantic.Similarity)) + geom_histogram(bins = 30, fill = "steelblue", color = "black") + labs(title = "Semantic Similarity", x = "Semantic Similarity", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
vid_dur <- ggplot(data_clean, aes(x = Video.Duration.Sec.Log)) + geom_histogram(bins = 30, fill = "steelblue", color = "black") + labs(title = "Video Length", x = "Log of Duration [s]", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
vid_sub <- ggplot(data_clean, aes(x = Subscriber.Count.Log)) + geom_histogram(bins = 30, fill = "steelblue", color = "black") + labs(title = "Subscribers", x = "Log of Subscribers Count", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
vid_title_lenght <- ggplot(data_clean, aes(x = Video.Title.Length)) + geom_histogram(bins = 30, fill = "steelblue", color = "black") + labs(title = "Video Title Length", x = "Characters in Title", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 16, hjust = 0.5)) 
grid.arrange(vid_color, vid_sim, vid_dur,vid_sub,vid_title_lenght, ncol=2, nrow=3)

# PLOT ALL DEPENDENT VARIABLES
vid_views_log <- ggplot(data_clean, aes(x = Video.Views)) + geom_histogram(bins = 30, fill = "steelblue", color = "black") + labs(title = "Video Views", x = "Views", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 14, hjust = 0.5)) 
vid_Shallow_eng <- ggplot(data_clean, aes(x = Video.Shallow.Engagement)) + geom_histogram(bins = 30, fill = "steelblue", color = "black") + labs(title = "Shallow Engagement", x = "Shallow Engagement", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 14, hjust = 0.5)) 
vid_deep_eng <- ggplot(data_clean, aes(x = Video.Deep.Engagement)) + geom_histogram(bins = 30, fill = "steelblue", color = "black") + labs(title = "Deep Engagement", x = "Deep Engagement", y = "Video Count") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 12), axis.text.y = element_text(size = 12), axis.title = element_text(size = 14), plot.title = element_text(size = 14, hjust = 0.5)) 
grid.arrange(vid_views_log, vid_Shallow_eng, vid_deep_eng, ncol=3, nrow=1)


#### 02 | CORRELATION ANALYSIS -------------------------------------------------

# ALL AS TABLE
calculate_correlations <- function(data, target_variables, other_variables, method = "pearson") {
  # Check if the method is valid
  if(!method %in% c("pearson", "kendall", "spearman")) {
    stop("Invalid method. Choose 'pearson', 'kendall', or 'spearman'.")
  }
  
  # Initialize an empty list to store correlation results
  correlation_results <- list()
  
  # Calculate correlations for each target variable
  for (target_variable in target_variables) {
    correlations <- numeric(length(other_variables))
    for (i in seq_along(other_variables)) {
      correlations[i] <- cor(data[[target_variable]], data[[other_variables[i]]], method = method)
    }
    # Round the correlation values to 3 decimal places
    correlations <- round(correlations, 3)
    correlation_results[[target_variable]] <- correlations
  }
  
  # Combine the results into a data frame
  correlation_table <- do.call(rbind, correlation_results)
  colnames(correlation_table) <- other_variables
  rownames(correlation_table) <- target_variables
  
  return(correlation_table)
}

all_target_variables <- c("Video.Views.Log", "Video.Shallow.Engagement.CubicRoot", "Video.Deep.Engagement.EighthRoot")
all_continuous_columns <- c("Subscriber.Count.Log", "Colorfulness.Score.SquareRoot", "Semantic.Similarity","Video.Title.Length", "Video.Duration.Sec.Log")

result <- calculate_correlations(data_clean, all_target_variables, all_continuous_columns)
print(result)


# PLOTS FOR
create_scatterplots <- function(data, target_variable, other_variables, aliases, method = "pearson") {
  # Check if the method is valid
  if(!method %in% c("pearson", "kendall", "spearman")) {
    stop("Invalid method. Choose 'pearson', 'kendall', or 'spearman'.")
  }
  
  # Check if aliases vector has the correct length
  if(length(aliases) != length(c(target_variable, other_variables))) {
    stop("The length of the aliases vector must match the length of the target and other variables combined.")
  }
  
  # Extract the alias names
  target_alias <- aliases[1]
  other_aliases <- aliases[-1]
  
  # Initialize an empty list to store the plots
  plot_list <- list()
  
  # Create scatterplots for the target variable against each other variable
  for (i in seq_along(other_variables)) {
    other_variable <- other_variables[i]
    other_alias <- other_aliases[i]
    
    # Calculate the correlation
    correlation <- cor(data[[target_variable]], data[[other_variable]], method = method)
    correlation <- round(correlation, 3)
    
    # Create the scatterplot
    p <- ggplot(data, aes_string(x = other_variable, y = target_variable)) +
      geom_point(color = "steelblue", alpha = 0.6) +
      labs(subtitle = paste("Corr:", correlation),
           x = other_alias) +
      theme_minimal() +
      theme(
        plot.subtitle = element_text(size = 14),
        axis.title.x = element_text(size = 14, face = "bold"),  # Increase x-axis title
        axis.title.y = element_blank(),  # Remove y-axis title
        axis.text = element_text(size = 12)                    # Increase axis text
      )
    
    # Add the plot to the list
    plot_list[[other_variable]] <- p
  }
  
  # Create the overall title
  overall_title <- textGrob(paste("Correlation of", target_alias, "against others"), gp = gpar(fontsize = 18, fontface = "bold"))
  
  # Arrange the plots in a grid with 2 columns and 3 rows, add the overall title with extra spacing
  grid.arrange(
    arrangeGrob(overall_title, padding = unit(2, "lines")),  # Increase padding for more spacing
    arrangeGrob(grobs = plot_list, ncol = 3, nrow = 2),
    heights = unit.c(unit(3, "lines"), unit(1, "npc") - unit(3, "lines"))  # Adjust heights to increase space
  )
}

#VIEWS
target_variable <- "Video.Views.Log"
aliases <- c("Video Views", "Subscriber Count", "Colorfulness Score", "Semantic Similarity", "Video Title Length", "Video Duration")

create_scatterplots(data_clean, target_variable, all_continuous_columns, aliases)

#Shallow
target_variable <- "Video.Shallow.Engagement.CubicRoot"
aliases <- c("Shallow Engagement", "Subscriber Count", "Colorfulness Score", "Semantic Similarity", "Video Title Length", "Video Duration")

create_scatterplots(data_clean, target_variable, all_continuous_columns, aliases)

#DEEP
target_variable <- "Video.Deep.Engagement.EighthRoot"
aliases <- c("Deep Engagement", "Subscriber Count", "Colorfulness Score", "Semantic Similarity", "Video Title Length", "Video Duration")

create_scatterplots(data_clean, target_variable, all_continuous_columns, aliases)


#### 03 | IMPACT CATEGORICAL VARIABLES - ANOVA -----------------------------------

# Function to perform ANOVA and calculate Eta Squared with enhanced indicators
calculate_anova_eta <- function(data, dependent_var, categorical_vars) {
  results <- data.frame(
    Variable = character(),
    F_Value = numeric(),
    P_Value = numeric(),
    Eta_Squared = numeric(),
    F_Value_Interpretation = character(),
    Significant_Impact = character(),
    Strength_of_Impact = character(),
    stringsAsFactors = FALSE
  )
  
  for (var in categorical_vars) {
    # Perform ANOVA
    anova_model <- aov(reformulate(var, response = dependent_var), data = data)
    anova_summary <- summary(anova_model)
    
    # Extract ANOVA results
    SSB <- anova_summary[[1]][var, "Sum Sq"]
    SSW <- anova_summary[[1]]["Residuals", "Sum Sq"]
    SST <- SSB + SSW
    eta_sq <- round(SSB / SST,5)
    
    F_value <- round(anova_summary[[1]][var, "F value"],2)
    P_value <- anova_summary[[1]][var, "Pr(>F)"]
    
    # Interpret the F-value
    if (F_value > 100) {
      f_value_interpretation <- "extremely strong"
    } else if (F_value > 50) {
      f_value_interpretation <- "very strong"
    } else if (F_value > 10) {
      f_value_interpretation <- "strong"
    } else if (F_value > 1) {
      f_value_interpretation <- "moderate"
    } else {
      f_value_interpretation <- "weak"
    }
    
    # Determine the significance impact
    if (P_value < 0.001) {
      significant_impact <- "very high"
    } else if (P_value < 0.01) {
      significant_impact <- "high"
    } else if (P_value < 0.05) {
      significant_impact <- "moderate"
    } else {
      significant_impact <- "none"
    }
    
    # Determine the strength of impact
    if (eta_sq >= 0.14) {
      strength_of_impact <- "large"
    } else if (eta_sq >= 0.06) {
      strength_of_impact <- "medium"
    } else {
      strength_of_impact <- "small"
    }
    
    # Append results to dataframe
    results <- rbind(results, data.frame(
      Variable = var,
      F_Value = F_value,
      P_Value = P_value,
      Eta_Squared = eta_sq,
      F_Value_Interpretation = f_value_interpretation,
      Significant_Impact = significant_impact,
      Strength_of_Impact = strength_of_impact
    ))
  }
  
  return(results)
}

# Categorical variables 
categorical_columns <- c("Category.ID", "TN.Human.Faces.Emotion", 
                         "TN.Human.Faces.Gender", "TN.Human.Faces", 
                         "TN.Logos", "TN.Text", "Video.Title.Emoji", 
                         "Video.Title.Sentiment.Vader.Cluster")

# Usage of the function
anova_results_view <- calculate_anova_eta(data_clean, "Video.Views.Log", categorical_columns)
anova_results_view

anova_results_Shallow <- calculate_anova_eta(data_clean, "Video.Shallow.Engagement.CubicRoot", categorical_columns)
anova_results_Shallow

anova_results_deep <- calculate_anova_eta(data_clean, "Video.Deep.Engagement.EighthRoot", categorical_columns)
anova_results_deep


#### 04 | IMPACT CATEGORICAL VARIABLES - PLOTS --------------------------------------

# Function to create a boxplot for each categorical variable
create_boxplot <- function(data, dependent_var, categorical_var, var_names = NULL, custom_titles = NULL, title_size = 14, axis_title_size = 12, axis_text_size = 10, show_x_axis_title = TRUE) {
  dependent_var_sym <- sym(dependent_var)
  categorical_var_sym <- sym(categorical_var)
  
  if (is.null(var_names)) {
    x_label <- rlang::as_string(categorical_var_sym)
    y_label <- rlang::as_string(dependent_var_sym)
  } else {
    x_label <- var_names[[categorical_var]]
    y_label <- var_names[[dependent_var]]
  }
  
  if (is.null(custom_titles)) {
    plot_title <- paste(y_label, "by", x_label)
  } else {
    plot_title <- custom_titles[[categorical_var]]
  }
  
  plot <- ggplot(data %>% 
                   group_by(!!categorical_var_sym) %>% 
                   mutate(Median = median(!!dependent_var_sym, na.rm = TRUE)) %>% 
                   ungroup() %>% 
                   mutate(!!categorical_var_sym := reorder(!!categorical_var_sym, Median, FUN = median)),
                 aes(x = !!categorical_var_sym, y = !!dependent_var_sym)) +
    geom_boxplot() +
    labs(title = plot_title, x = if (show_x_axis_title) x_label else NULL, y = y_label) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = axis_text_size),
      axis.text.y = element_text(size = axis_text_size),
      axis.title.x = if (show_x_axis_title) element_text(size = axis_title_size) else element_blank(),
      axis.title.y = element_text(size = axis_title_size),
      plot.title = element_text(size = title_size, hjust = 0.5)
    )
  return(plot)
}

# Function to plot and arrange all categorical variables
plot_and_arrange_all <- function(data, dependent_var, categorical_vars, var_names = NULL, custom_titles = NULL, title_size = 14, axis_title_size = 12, axis_text_size = 10, show_x_axis_title = TRUE) {
  plots <- list()
  
  for (var in categorical_vars) {
    plots[[var]] <- create_boxplot(data, dependent_var, var, var_names, custom_titles, title_size, axis_title_size, axis_text_size, show_x_axis_title)
  }
  
  # Combine all plots into a grid
  do.call(gridExtra::grid.arrange, c(plots, ncol = 3, nrow = ceiling(length(plots) / 3)))
}

# Example usage
categorical_columns <- c("Category.ID", "TN.Human.Faces.Emotion", 
                         "TN.Human.Faces.Gender", "TN.Human.Faces", 
                         "TN.Logos", "TN.Text", "Video.Title.Emoji", 
                         "Video.Title.Sentiment.Vader.Cluster")

var_names <- list(
  "Video.Views.Log" = "Log of Video Views",
  "Video.Shallow.Engagement.CubicRoot" = "Cubic Root Shallow",
  "Video.Deep.Engagement.EighthRoot" = "Eighth Root Deep",
  "Category.ID" = "Video Category",
  "TN.Human.Faces" = "Face",
  "TN.Human.Faces.Emotion" = "Emotion",
  "TN.Human.Faces.Gender" = "Gender",
  "TN.Logos" = "Logo",
  "TN.Text" = "Text",
  "Video.Title.Emoji" = "Title Emoji",
  "Video.Title.Sentiment.Vader.Cluster" = "Title Sentiment"
)

custom_titles <- list(
  "Category.ID" = "Views by Video Category",
  "TN.Human.Faces" = "Views by Face Presence",
  "TN.Human.Faces.Emotion" = "Views by Emotion",
  "TN.Human.Faces.Gender" = "Views by Gender",
  "TN.Logos" = "Views by Logo Presence",
  "TN.Text" = "Views by Text Presence",
  "Video.Title.Emoji" = "Views by Presence of Emoji in Title",
  "Video.Title.Sentiment.Vader.Cluster" = "Views by Title Sentiment"
)

# Assuming 'data_clean' is your dataframe and 'Video.Views.Log' is the dependent variable
plot_and_arrange_all(data_clean, "Video.Views.Log", categorical_columns, var_names, custom_titles, title_size = 12, axis_title_size = 11, axis_text_size = 10, show_x_axis_title = FALSE)
plot_and_arrange_all(data_clean, "Video.Shallow.Engagement.CubicRoot", categorical_columns, var_names, custom_titles, title_size = 12, axis_title_size = 11, axis_text_size = 10, show_x_axis_title = FALSE)
plot_and_arrange_all(data_clean, "Video.Deep.Engagement.EighthRoot", categorical_columns, var_names, custom_titles, title_size = 12, axis_title_size = 11, axis_text_size = 10, show_x_axis_title = FALSE)


#### 05 | LINEAR REGRESSION - GET BEST MODEL --------------------------------------

# From full model to best model
full_model_views <- lm(Video.Views.Log ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + 
                   TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
                   Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
                 data = data_clean)

full_model_Shallow <- lm(Video.Shallow.Engagement.CubicRoot ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + 
                         TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
                         Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
                       data = data_clean)

full_model_deep <- lm(Video.Deep.Engagement.EighthRoot ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + 
                         TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
                         Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
                       data = data_clean)

options(na.action='na.fail')

# For VIEWS
model_set_views <- dredge(full_model_views)
top_models_views <- model_set_views[order(model_set_views$AIC)]
View(top_models_views) # Check the top models based on AIC

# For Shallow Engagement
model_set_Shallow <- dredge(full_model_Shallow)
top_models_Shallow <- model_set_Shallow[order(model_set_Shallow$AIC)]
View(top_models_Shallow) # Check the top models based on AIC
summary(top_models_Shallow)

# For Deep Engagement
model_set_deep <- dredge(full_model_deep)
top_models_deep <- model_set_deep[order(model_set_deep$AIC)]
View(top_models_deep) # Check the top models based on AIC
summary(top_models_deep)


# MAIN FOUND MODELS based on dredge

# For VIEWS
video_views_log = lm(Video.Views.Log ~  Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + TN.Human.Faces + TN.Human.Faces.Emotion + 
                       TN.Logos + TN.Text + Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, data = data_clean)
vif(video_views_log)

#shows that although dredge says TN.Human.Faces should be contained. It is an aligned variable with TN.Human.Face.Emotions
video_views_log = lm(Video.Views.Log ~  Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + TN.Human.Faces.Emotion + 
                       TN.Logos + TN.Text + Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

summary(video_views_log)
par(mfrow=c(2, 2))
plot(video_views_log)


# For Shallow Engagement
video_Shallow_cubic = lm(Video.Shallow.Engagement.CubicRoot ~  Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + TN.Human.Faces.Gender + TN.Human.Faces.Emotion + 
                       TN.Logos + TN.Text + Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, data = data_clean)
vif(video_Shallow_cubic) # seems that there are aliased variables

video_Shallow_cubic = lm(Video.Shallow.Engagement.CubicRoot ~  Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length  + TN.Human.Faces.Emotion + 
                          TN.Logos + TN.Text + Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, data = data_clean)
vif(video_Shallow_cubic)
summary(video_Shallow_cubic)
par(mfrow=c(2, 2))
plot(video_Shallow_cubic)


# For Deep Engagement
video_deep_r8 = lm(Video.Deep.Engagement.EighthRoot ~  Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity  + TN.Human.Faces.Gender + TN.Human.Faces.Emotion + 
                       TN.Logos + TN.Text + Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, data = data_clean)
vif(video_deep_r8) # seems that there are aliased variables

video_deep_r8 = lm(Video.Deep.Engagement.EighthRoot ~  Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + TN.Human.Faces.Emotion + 
                     TN.Logos + TN.Text + Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, data = data_clean)
vif(video_deep_r8)
summary(video_deep_r8)
par(mfrow=c(2, 2))
plot(video_deep_r8)


#### 06 | LINEAR REGRESSION - CHECK INTERACTIONS --------------------------------------

# For Views
video_views_log_inter_dur_emoji = lm(Video.Views.Log ~  Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + TN.Human.Faces.Emotion + 
                       TN.Logos + TN.Text + Video.Title.Emoji * Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

video_views_log_inter_sim_vader = lm(Video.Views.Log ~  Category.ID + TN.Human.Faces.Emotion + Colorfulness.Score.SquareRoot + Semantic.Similarity * Video.Title.Sentiment.Vader.Cluster + Video.Title.Emoji + Video.Title.Length +
                       TN.Logos + TN.Text + Video.Duration.Sec.Log, data = data_clean)

video_views_log_inter_col_text = lm(Video.Views.Log ~  Category.ID + TN.Human.Faces.Emotion + Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster + Video.Title.Emoji + Video.Title.Length +
                                       TN.Logos + TN.Text * Colorfulness.Score.SquareRoot + Video.Duration.Sec.Log, data = data_clean)

video_views_log_w_sub_inter_col_sub = lm(Video.Views.Log ~  Category.ID + Semantic.Similarity + Video.Title.Length + TN.Human.Faces.Emotion + 
                             TN.Logos + TN.Text + Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster + Subscriber.Count.Log * Colorfulness.Score.SquareRoot , data = data_clean)

summary(video_views_log_inter_dur_emoji) # Indicates Significance
summary(video_views_log_inter_sim_vader) # No Significance
summary(video_views_log_inter_col_text) # Indicates Significance
summary(video_views_log_w_sub_inter_col_sub) # Indicates Significance


# For Shallow Engagement
Shallow_cubic_inter_dur_emoji = lm(Video.Shallow.Engagement.CubicRoot ~  Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + TN.Human.Faces.Emotion + 
                                       TN.Logos + TN.Text + Video.Title.Emoji * Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

Shallow_cubic_inter_sim_vader = lm(Video.Shallow.Engagement.CubicRoot ~  Category.ID + TN.Human.Faces.Emotion + Colorfulness.Score.SquareRoot + Semantic.Similarity * Video.Title.Sentiment.Vader.Cluster + Video.Title.Emoji + Video.Title.Length +
                                       TN.Logos + TN.Text + Video.Duration.Sec.Log, data = data_clean)

Shallow_cubic_inter_col_text = lm(Video.Shallow.Engagement.CubicRoot ~  Category.ID + TN.Human.Faces.Emotion + Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster + Video.Title.Emoji + Video.Title.Length +
                                      TN.Logos + TN.Text * Colorfulness.Score.SquareRoot + Video.Duration.Sec.Log, data = data_clean)

Shallow_cubic_w_sub_inter_col_sub = lm(Video.Shallow.Engagement.CubicRoot ~  Category.ID + Semantic.Similarity + Video.Title.Length + TN.Human.Faces.Emotion + 
                                           TN.Logos + TN.Text + Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster + Subscriber.Count.Log * Colorfulness.Score.SquareRoot , data = data_clean)

summary(Shallow_cubic_inter_dur_emoji) # Indicates significance
summary(Shallow_cubic_inter_sim_vader) # No Significance
summary(Shallow_cubic_inter_col_text) # No Significance
summary(Shallow_cubic_w_sub_inter_col_sub) # Indicates significance


# For Deep Engagement
deep_r8_inter_dur_emoji = lm(Video.Deep.Engagement.EighthRoot ~  Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + TN.Human.Faces.Emotion + 
                                    TN.Logos + TN.Text + Video.Title.Emoji * Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

deep_r8_inter_sim_vader = lm(Video.Deep.Engagement.EighthRoot ~  Category.ID + TN.Human.Faces.Emotion + Colorfulness.Score.SquareRoot + Semantic.Similarity * Video.Title.Sentiment.Vader.Cluster + Video.Title.Emoji +
                                    TN.Logos + TN.Text + Video.Duration.Sec.Log, data = data_clean)

deep_r8_inter_col_text = lm(Video.Deep.Engagement.EighthRoot ~  Category.ID + TN.Human.Faces.Emotion + Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster + Video.Title.Emoji +
                                   TN.Logos + TN.Text * Colorfulness.Score.SquareRoot + Video.Duration.Sec.Log, data = data_clean)

deep_r8_w_sub_inter_col_sub = lm(Video.Deep.Engagement.EighthRoot ~  Category.ID + Semantic.Similarity + TN.Human.Faces.Emotion + 
                                        TN.Logos + TN.Text + Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster + Subscriber.Count.Log * Colorfulness.Score.SquareRoot , data = data_clean)

summary(deep_r8_inter_dur_emoji) # No Significance
summary(deep_r8_inter_sim_vader) # No Significance
summary(deep_r8_inter_col_text) # Indicates significance
summary(deep_r8_w_sub_inter_col_sub) # Indicates significance


# New Models with all interactions

# For Views
'Explain interaction: While "text boxes" in the thumbnail on their own are associated with more views, 
their presence reduces the additional positive effect that increased colorfulness would have had. 
This might be because text boxes can distract from the visual appeal of the colors.

While emojis in the title are associated with more views, their positive effect decreases as the video duration increases. 
This could be because longer videos might be perceived as more serious or informative, and the presence of 
emojis could make them seem less so, thereby reducing the positive impact.'
video_views_log_all_inter = lm(Video.Views.Log ~  Category.ID + Colorfulness.Score.SquareRoot * TN.Text + Video.Title.Length + TN.Human.Faces.Emotion + 
                       TN.Logos + Video.Title.Emoji * Video.Duration.Sec.Log + Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

summary(video_views_log_all_inter)
par(mfrow=c(2, 2))
plot(video_views_log_all_inter)


# For Shallow
'Explain the interaction: Longer video seem to result in slightly less Shallow engagement (per view), as idicated by the negative coefficient for Log.Duraation.
Additional the positive effect on Shallow engagement by emojis in the title is also reduced with longer videos.'
video_Shallow_cubic_all_inter = lm(Video.Shallow.Engagement.CubicRoot ~  Category.ID + Colorfulness.Score.SquareRoot + TN.Text + Video.Title.Length + TN.Human.Faces.Emotion + 
                                 TN.Logos + Video.Title.Emoji * Video.Duration.Sec.Log + Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

summary(video_Shallow_cubic_all_inter)
par(mfrow=c(2, 2))
plot(video_Shallow_cubic_all_inter)


# For Deep
'When there is no text box in the thumbnail, an increase in the colorfulness score is associated with a decrease in deep engagement.

The presence of text boxes in the thumbnail on its own has a small negative effect on deep engagement, but this effect is not 
statistically significant in this model.

The interaction term suggests that while increased colorfulness typically reduces deep engagement, 
the presence of text boxes in the thumbnail mitigates this negative effect. 
This could indicate that text boxes in a colorful thumbnail provide useful context or information that helps retain deep engagement despite 
the "distraction of bright colors".'
video_deep_r8_all_inter = lm(Video.Deep.Engagement.EighthRoot ~  Category.ID + Colorfulness.Score.SquareRoot * TN.Text + TN.Human.Faces.Emotion + 
                                    TN.Logos + Video.Title.Emoji + Video.Duration.Sec.Log + Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

summary(video_deep_r8_all_inter)
par(mfrow=c(2, 2))
plot(video_deep_r8_all_inter)


#### 07 | LINEAR REGRESSION - MODEL DIAGNOSTICS --------------------------------------

# Define the function to create diagnostic plots with larger text and custom titles
create_diagnostic_plots <- function(model) {
  # Set up the plotting area
  par(mfrow = c(2, 2), mar = c(5, 5, 4, 2) + 0.1, oma = c(0, 0, 2, 0))
  
  # Residuals vs Fitted plot
  plot(model, which = 1, 
       main = "",       # Suppress the automatic title
       caption = "",    # Suppress the automatic caption
       sub.caption = "",  # Suppress the default sub.caption
       cex.lab = 1.5,   # Increase axis label size
       cex.axis = 1.5)  # Increase axis tick size
  title(main = "Residuals vs Fitted", cex.main = 1.8)  # Custom title with increased size
  
  # Q-Q plot
  plot(model, which = 2, 
       main = "", 
       caption = "",
       sub.caption = "", 
       cex.lab = 1.5, 
       cex.axis = 1.5)
  title(main = "Normal Q-Q", cex.main = 1.8)  # Custom title with increased size
  
  # Scale-Location plot
  plot(model, which = 3, 
       main = "", 
       caption = "",
       sub.caption = "", 
       cex.lab = 1.5, 
       cex.axis = 1.5)
  title(main = "Scale-Location", cex.main = 1.8)  # Custom title with increased size
  
  # Residuals vs Leverage plot with Cook's distance
  plot(model, which = 5, 
       main = "", 
       caption = "",
       sub.caption = "", 
       cex.lab = 1.5, 
       cex.axis = 1.5,
       cook.levels = 4 / (nrow(model$model) - length(coef(model))))
  title(main = "Residuals vs Leverage", cex.main = 1.8)  # Custom title with increased size
  
  # Reset the plotting area to default
  par(mfrow = c(1, 1))
}

# VIEWS
model1 <- lm(Video.Views.Log ~ Category.ID + Colorfulness.Score.SquareRoot * TN.Text + Video.Title.Length + 
               TN.Human.Faces.Emotion + TN.Logos + Video.Title.Emoji * Video.Duration.Sec.Log + 
               Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

create_diagnostic_plots(model1)
summary(model1)


# SHALLOW
model2 <- lm(Video.Shallow.Engagement.CubicRoot ~ Category.ID + Colorfulness.Score.SquareRoot + TN.Text + Video.Title.Length + TN.Human.Faces.Emotion + 
               TN.Logos + Video.Title.Emoji * Video.Duration.Sec.Log + Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

create_diagnostic_plots(model2)
summary(model2)


#DEEP
model3 <- lm(Video.Deep.Engagement.EighthRoot ~ Category.ID + Colorfulness.Score.SquareRoot * TN.Text + TN.Human.Faces.Emotion + 
               TN.Logos + Video.Title.Emoji + Video.Duration.Sec.Log + Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

create_diagnostic_plots(model3)
summary(model3)


#### 07 | LINEAR REGRESSION - NON-LINEAR RELATION - GLM --------------------------------------

video_views_glm = glm(Video.Views.Log ~ Category.ID + 
                        Colorfulness.Score.SquareRoot * TN.Text + 
                        Video.Title.Length + TN.Human.Faces.Emotion + 
                        TN.Logos + Video.Title.Emoji * Video.Duration.Sec.Log + 
                        Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster, 
                      family = Gamma(link = "log"), data = data_clean)

create_diagnostic_plots(video_views_glm)
create_diagnostic_plots(model1)

summary(video_views_glm)
summary(model1)

AIC(video_views_glm)
AIC(model1)


#### 08 | LINEAR REGRESSION - ROBUST REGRESSION (OUTLIERS & LEVERAGE) --------------------------------------

# Train Both Models again on train
video_views_log_all_inter = lm(Video.Views.Log ~  Category.ID + Colorfulness.Score.SquareRoot * TN.Text + Video.Title.Length + TN.Human.Faces.Emotion + 
                                 TN.Logos + Video.Title.Emoji * Video.Duration.Sec.Log + Semantic.Similarity + Video.Title.Sentiment.Vader.Cluster, data = data_clean)

robust_model <- rlm(Video.Views.Log ~ Category.ID + Colorfulness.Score.SquareRoot * TN.Text + 
                      Video.Title.Length + TN.Human.Faces.Emotion + TN.Logos + 
                      Video.Title.Emoji * Video.Duration.Sec.Log + Semantic.Similarity + 
                      Video.Title.Sentiment.Vader.Cluster, 
                    data = data_clean, psi = psi.huber, k = 1.345)

summary(video_views_log_all_inter)
summary(robust_model)

create_diagnostic_plots(video_views_log_all_inter)
create_diagnostic_plots(robust_model)

AIC(video_views_log_all_inter)
AIC(robust_model)

'Very similar. Now lets also look at the predicted values.' 


# Direct Evaluation (Holdout Method) Performance Check ####
predictions_lm <- predict(video_views_log_all_inter, test_data)
predictions_rlm <- predict(robust_model, test_data)


# Compute residuals on original scope
residuals_lm <- test_data$Video.Views - exp(predictions_lm)
residuals_rlm <- test_data$Video.Views - exp(predictions_rlm)


# Calculate MAE and RMSE manually
mae_lm <- mean(abs(residuals_lm))
rmse_lm <- sqrt(mean(residuals_lm^2))
mae_rlm <- mean(abs(residuals_rlm))
rmse_rlm <- sqrt(mean(residuals_rlm^2))

# Print MAE and RMSE
print(paste("Mean Absolute Error: ", mae_lm))
print(paste("Root Mean Squared Error: ", rmse_lm))
print(paste("Mean Absolute Error: ", mae_rlm))
print(paste("Root Mean Squared Error: ", rmse_rlm))

# Create a data frame for plotting
plot_data_lm <- data.frame(Actual = test_data$Video.Views, Predicted = exp(predictions_lm))
plot_data_rlm <- data.frame(Actual = test_data$Video.Views, Predicted = exp(predictions_rlm))

# Plot
ggplot(plot_data_lm, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +  # Scatter plot points
  geom_smooth(method = "lm", color = "blue", se = FALSE) +  # Linear regression line
  labs(x = "Actual Log of Video Views", y = "Predicted Log of Video Views",
       title = "Actual vs Predicted Log of Video Views") +
  ylim(0, 6.5e+05) +  # Set y-axis limit
  theme_minimal()

# Plot
ggplot(plot_data_rlm, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +  # Scatter plot points
  geom_smooth(method = "lm", color = "blue", se = FALSE) +  # Linear regression line
  labs(x = "Actual Log of Video Views", y = "Predicted Log of Video Views",
       title = "Actual vs Predicted Log of Video Views") +
  ylim(0, 6.5e+05) +  # Set y-axis limit
  theme_minimal()


'Very similar -> hence robust not required although leverage and outliers identified. Lets
also look a dataset that splits the outliers from the main dataset.'


#### 09 | LINEAR REGRESSION - SPLIT DATA APPORACH (OUTLIERS & LEVERAGE) --------------------------------------

# Calculate Cook's Distance
cooks_distances <- cooks.distance(video_views_log_all_inter) # video_views_log_all_inter, model_no_outliers, model_only_outliers

# Number of observations
n <- nrow(data_clean)

# Define threshold for high influence points
threshold <- 4/n

# Plot Cook's Distance
plot(cooks_distances, type='h', ylim=c(0, max(cooks_distances, na.rm = TRUE)), 
     main="Cook's Distance", ylab="Cook's Distance", xlab="Index")
abline(h=4/n, col="red", lty=2)  # Adding a horizontal line at 4/n

# Identify points with high Cook's distance
influential_points <- which(cooks_distances > threshold)

if(length(influential_points) > 0) {
  points(influential_points, cooks_distances[influential_points], col="red", pch=19)
}

length(influential_points) 
length(cooks_distances)

# Create subsets of datasets
data_no_outliers <- data_clean[-influential_points, ]
data_only_outliers <- data_clean[influential_points, ]

# Fit models on both datasets
model_no_outliers <- update(video_views_log_all_inter, data = data_no_outliers)
model_only_outliers <- update(video_views_log_all_inter, data = data_only_outliers)


compare_lm_models <- function(model1, model2, model3) {
  # Function to extract coefficients, values, and p-values
  extract_info <- function(model) {
    summary_model <- summary(model)
    data.frame(
      Coefficient = names(summary_model$coefficients[, 1]),
      Estimate = round(summary_model$coefficients[, 1], 3),
      P_value = summary_model$coefficients[, 4]
    )
  }
  
  # Extract information from each model
  model1_info <- extract_info(model1)
  model2_info <- extract_info(model2)
  model3_info <- extract_info(model3)
  
  # Merge the data frames by Coefficient
  merged_info <- Reduce(function(x, y) merge(x, y, by = "Coefficient", all = TRUE),
                        list(model1_info, model2_info, model3_info))
  
  # Rename columns for clarity
  colnames(merged_info) <- c("Coefficient",
                             "Estimate_Model1", "P_value_Model1",
                             "Estimate_Model2", "P_value_Model2",
                             "Estimate_Model3", "P_value_Model3")
  
  # Function to categorize p-values
  categorize_p_value <- function(p_value) {
    if (is.na(p_value)) {
      return(NA)
    } else if (p_value < 0.001) {
      return("significance level 3")
    } else if (p_value < 0.01) {
      return("significance level 2")
    } else if (p_value < 0.05) {
      return("significance level 1")
    } else {
      return("not significant")
    }
  }
  
  # Apply categorization to all p-values
  merged_info$P_value_Model1_Category <- sapply(merged_info$P_value_Model1, categorize_p_value)
  merged_info$P_value_Model2_Category <- sapply(merged_info$P_value_Model2, categorize_p_value)
  merged_info$P_value_Model3_Category <- sapply(merged_info$P_value_Model3, categorize_p_value)
  
  # Evaluate changes in direction for estimates and significance levels for p-values
  evaluate_change <- function(estimate1, estimate2, p_value1_cat, p_value2_cat) {
    # Determine change in estimate direction
    if (is.na(estimate1) || is.na(estimate2)) {
      estimate_change <- NA
    } else if (sign(estimate1) != sign(estimate2)) {
      estimate_change <- "diff"
    } else {
      estimate_change <- "-"
    }
    
    # Determine change in significance level
    if (is.na(p_value1_cat) || is.na(p_value2_cat)) {
      p_value_change <- NA
    } else if (p_value1_cat != p_value2_cat) {
      levels <- c("not significant", "significance level 1", "significance level 2", "significance level 3")
      p_value1_level <- match(p_value1_cat, levels)
      p_value2_level <- match(p_value2_cat, levels)
      if (p_value1_level > p_value2_level) {
        p_value_change <- "lower"
      } else {
        p_value_change <- "higher"
      }
    } else {
      p_value_change <- "-"
    }
    
    return(c(estimate_change, p_value_change))
  }
  
  # Apply the evaluation function to model 2 and model 3
  model2_changes <- mapply(evaluate_change,
                           merged_info$Estimate_Model1, merged_info$Estimate_Model2,
                           merged_info$P_value_Model1_Category, merged_info$P_value_Model2_Category,
                           SIMPLIFY = FALSE)
  
  model3_changes <- mapply(evaluate_change,
                           merged_info$Estimate_Model1, merged_info$Estimate_Model3,
                           merged_info$P_value_Model1_Category, merged_info$P_value_Model3_Category,
                           SIMPLIFY = FALSE)
  
  # Add changes to the data frame
  merged_info$Estimate_Model2 <- sapply(model2_changes, `[[`, 1)
  merged_info$P_value_Model2 <- sapply(model2_changes, `[[`, 2)
  
  merged_info$Estimate_Model3 <- sapply(model3_changes, `[[`, 1)
  merged_info$P_value_Model3 <- sapply(model3_changes, `[[`, 2)
  
  # Replace the p-values with their categories
  merged_info$P_value_Model1 <- merged_info$P_value_Model1_Category
  merged_info$P_value_Model2 <- merged_info$P_value_Model2
  merged_info$P_value_Model3 <- merged_info$P_value_Model3
  
  # Remove the category columns
  merged_info <- merged_info[, -grep("Category", names(merged_info))]
  
  # Reorder the columns
  merged_info <- merged_info[, c("Coefficient",
                                 "Estimate_Model1", "Estimate_Model2", "Estimate_Model3",
                                 "P_value_Model1", "P_value_Model2", "P_value_Model3")]
  
  return(merged_info)
}

# Example usage of the function
model_comparison_wide <- compare_lm_models(video_views_log_all_inter, model_no_outliers, model_only_outliers)
print(model_comparison_wide)
View(model_comparison_wide)

# Output the wide format comparison table
View(model_comparison_wide)

summary(video_views_log_all_inter)
summary(model_no_outliers)
summary(model_only_outliers)

AIC(video_views_log_all_inter)
AIC(model_no_outliers)
AIC(model_only_outliers)

'Major coefficients seem to stay but significance differs in the outlier group.'


#### 10 | RANDOM FOREST - VIEWS --------------------------------------

'1. FIND OPTIMAL NUMBER OF TRESS'
set.seed(123)  # for reproducibility

# Find ideal number of trees when model error stableizes
tree_numbers <- c(50, 100, 200, 500, 1000)

# Initialize vectors to store models and their MSE
models <- list()
oob_mse <- numeric(length(tree_numbers))

# Loop through specified 'ntree' values
for (i in seq_along(tree_numbers)) {
  set.seed(123)  # for reproducibility
  models[[i]] <- randomForest(Video.Shallow.Engagement.CubicRoot ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + 
                                TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
                                Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
                              data = data_clean, ntree = tree_numbers[i])
  oob_mse[i] <- tail(models[[i]]$mse, 1)  # Fetch the last MSE value from each model
}

# Plotting MSE against number of trees
plot(tree_numbers, oob_mse, type = "b", pch = 19, col = "blue",
     xlab = "Number of Trees", ylab = "Out-of-Bag MSE",
     main = "OOB MSE vs. Number of Trees",
     cex.main = 1.7,
     cex.lab = 1.6,  # Increase x and y axis labels text size
     cex.axis = 1.4  # Increase x and y axis ticks text size
)

#Old results. 
#oob_mse = c(3.809475, 3.721470, 3.683022, 3.667255, 3.661145) 
'From the results it seems like 500 is the last major improvement without beeing overly heavly in the computation' 


'2. HYPERPARAMETER TUNING'
set.seed(123)  
# Define training control
train_control <- trainControl(
  method = "cv",        # cross-validation
  number = 10,          # number of folds
  verboseIter = TRUE,   # show training progress
  search = "grid"       # grid search (can also use "random" for random search)
)

# Define the tuning grid
tune_grid <- expand.grid(
  mtry = seq(3, 11, 1),
  splitrule = "variance",
  min.node.size = c(5, 10, 15, 20)
)

# Train the model
model_caret <- train(
  Video.Views.Log ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + 
    TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
    Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
  data = data_clean,
  method = "ranger",  # Specifies the ranger model
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "RMSE",     # Metric to use for model selection
  num.trees = 500,
  importance = 'impurity' # Calculate variable importance based on impurity
)

# Print the model results
print(model_caret)

'Best model chosen based on smallest RMSE value at 

mtry = 7
min.node.size = 15 

RMSE:1.911753
Rsquared: 0.1171262
MAE: 1.563811'

importance <- varImp(model_caret, scale = TRUE)
importance_df <- as.data.frame(importance$importance)
importance_df <- data.frame(Variable = rownames(importance_df), Importance = importance_df$Overall)
importance_df <- importance_df[order(-importance_df$Importance), ]
# Print the sorted variable importance data frame
print(importance_df)


#### 11 | RANDOM FOREST - SHALLOW ENGAGEMENT --------------------------------------

"1. Find optimal number of trees
2. Find optimal mtry"

'Take 500 trees as with the initial model for views. Shows to be also here the best but skip the testing.' 

'Train Control same as initially (see above)'

'Tune Grid same as initially (see above)'

# Train the model
model_caret_Shallow <- train(
  Video.Shallow.Engagement.CubicRoot ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + 
    TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
    Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
  data = data_clean,
  method = "ranger",  # Specifies the ranger model
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "RMSE",     # Metric to use for model selection
  num.trees = 500,
  importance = 'impurity' # Calculate variable importance based on impurity
)

# Print the model results
print(model_caret_Shallow)
'Best model chosen based on smallest RMSE value at 

mtry = 10
min.node.size = 15 

with 

RMSE:0.06578351
Rsquared: 0.2137991
MAE: 0.05195670'



# Get variable importance
importance <- varImp(model_caret_Shallow, scale = TRUE)
importance_df <- as.data.frame(importance$importance)
importance_df <- data.frame(Variable = rownames(importance_df), Importance = importance_df$Overall)
importance_df <- importance_df[order(-importance_df$Importance), ]
# Print the sorted variable importance data frame
print(importance_df)

'It seems that the following variables are important (decending order):

Video Duration
Semantic Similarity
Video Title Lenght
Colorfulness Score
Thumbnail Text
Thumbnail Logo
Category (Music)

The following seems to be a bit less important:
Face Gender
Face Emotion'



#### 12 | RANDOM FOREST - DEEP ENGAGEMENT --------------------------------------
"1. Find optimal number of trees
2. Find optimal mtry"

'Take 500 trees as with the initial model for views. Shows to be also here the best but skip the testing.' 

'Train Control same as initially (see above)'

'Tune Grid same as initially (see above)'
set.seed(123)

# Train the model
model_caret_deep <- train(
  Video.Deep.Engagement.EighthRoot ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length +
    TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
    Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
  data = data_clean,
  method = "ranger",  # Specifies the ranger model
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "RMSE",     # Metric to use for model selection
  num.trees = 500,
  importance = 'impurity' # Calculate variable importance based on impurity
)

# Print the model results
print(model_caret_deep)
'Best model chosen based on smallest RMSE value at 

mtry = 7
min.node.size = 20 

with 

RMSE:0.06054942
Rsquared: 0.1677321
MAE: 0.04801649'



# Get variable importance
importance <- varImp(model_caret_deep, scale = TRUE)
importance_df <- as.data.frame(importance$importance)
importance_df <- data.frame(Variable = rownames(importance_df), Importance = importance_df$Overall)
importance_df <- importance_df[order(-importance_df$Importance), ]
# Print the sorted variable importance data frame
print(importance_df)

'It seems that the following variables are important (decending order):

Video Duration
Semantic Similarity
Video Title Lenght
Colorfulness Score
Thumbnail Text
Thumbnail Logo
Category (Music)

The following seems to be a bit less important:
Face Gender
Face Emotion'


#### 13 | RANDOM FOREST - TRAIN BEST MODEL --------------------------------------

data_subset <- data_clean[, c("Video.Views.Log", "Category.ID", "Colorfulness.Score.SquareRoot", "Semantic.Similarity", 
                              "Video.Title.Length", "TN.Human.Faces", "TN.Human.Faces.Emotion", 
                              "TN.Human.Faces.Gender", "TN.Logos", "TN.Text", "Video.Title.Emoji", 
                              "Video.Duration.Sec.Log", "Video.Title.Sentiment.Vader.Cluster")]

# Set the seed for reproducibility
set.seed(123)


'# FOR VIEWS'
rf_model <- ranger(
  Video.Views.Log ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + 
    TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
    Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
  data = data_clean,
  num.trees = 500,
  mtry = 7,  # Example value, adjust based on tuning
  min.node.size = 15,  # Example value, adjust based on tuning
  importance = 'impurity'
)

# Print the model results
print(rf_model)

# Assuming rf_model is your trained ranger model
importance_values <- importance(rf_model)
importance_df <- data.frame(Variable = names(importance_values), Importance = as.numeric(importance_values))
importance_df <- importance_df[order(-importance_df$Importance), ]

print(importance_df)


'# FOR SHALLOW'
rf_model_shallow <- ranger(
  Video.Shallow.Engagement.CubicRoot ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + 
    TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
    Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
  data = data_clean,
  num.trees = 500,
  mtry = 10,  # Example value, adjust based on tuning
  min.node.size = 15,  # Example value, adjust based on tuning
  importance = 'impurity'
)

# Print the model results
print(rf_model_shallow)

# Assuming rf_model is your trained ranger model
importance_values_shallow <- importance(rf_model_shallow)
importance_df_shallow <- data.frame(Variable = names(importance_values_shallow), Importance = as.numeric(importance_values_shallow))
importance_df_shallow <- importance_df_shallow[order(-importance_df_shallow$Importance), ]

print(importance_df_shallow)


'# FOR DEEP'
rf_model_deep<- ranger(
  Video.Deep.Engagement.EighthRoot ~ Category.ID + Colorfulness.Score.SquareRoot + Semantic.Similarity + Video.Title.Length + 
    TN.Human.Faces + TN.Human.Faces.Emotion + TN.Human.Faces.Gender + TN.Logos + TN.Text + 
    Video.Title.Emoji + Video.Duration.Sec.Log + Video.Title.Sentiment.Vader.Cluster, 
  data = data_clean,
  num.trees = 500,
  mtry = 7,  # Example value, adjust based on tuning
  min.node.size = 20,  # Example value, adjust based on tuning
  importance = 'impurity'
)

# Print the model results
print(rf_model_deep)

# Assuming rf_model is your trained ranger model
importance_values_deep <- importance(rf_model_deep)
importance_df_deep <- data.frame(Variable = names(importance_values_deep), Importance = as.numeric(importance_values_deep))
importance_df_deep <- importance_df_deep[order(-importance_df_deep$Importance), ]

print(importance_df_deep)


#### 14 | RANDOM FOREST - GET ALL PLOTS --------------------------------------

# Function to generate PDPs for multiple variables with optional axes
generate_pdp_plots <- function(model, data, variables, var_types, custom_titles = NULL, ncol = 3, title_size = 12, axis_text_size = 10, point_size = 3, show_axes = TRUE) {
  
  # Check if custom_titles is provided and has the correct length
  if (is.null(custom_titles)) {
    custom_titles <- variables
  } else if (length(custom_titles) != length(variables)) {
    stop("Length of custom_titles must match length of variables")
  }
  
  # Helper function to customize plots
  customize_pdp <- function(pdp_data, var, var_type, title, show_axes) {
    base_plot <- ggplot(pdp_data, aes_string(x = var, y = "yhat")) +
      ggtitle(title) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = title_size, hjust = 0.5)
      )
    
    if (var_type == "categorical") {
      base_plot <- base_plot +
        geom_point(size = point_size)
      
      if (show_axes) {
        base_plot <- base_plot +
          theme(
            axis.text.x = element_text(size = axis_text_size, angle = 90, vjust = 0.5, hjust = 1),
            axis.text.y = element_text(size = axis_text_size),
            axis.title.x = element_blank(),
            axis.title.y = element_blank()
          )
      } else {
        base_plot <- base_plot +
          theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.x = element_blank(),
            axis.ticks.y = element_blank()
          )
      }
    } else {
      base_plot <- base_plot +
        geom_line(size = 1)
      
      if (show_axes) {
        base_plot <- base_plot +
          theme(
            axis.text.x = element_text(size = axis_text_size),
            axis.text.y = element_text(size = axis_text_size),
            axis.title.x = element_blank(),
            axis.title.y = element_blank()
          )
      } else {
        base_plot <- base_plot +
          theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.x = element_blank(),
            axis.ticks.y = element_blank()
          )
      }
    }
    
    return(base_plot)
  }
  
  # Generate the PDP plots
  pdp_plots <- lapply(seq_along(variables), function(i) {
    var <- variables[i]
    var_type <- var_types[i]
    title <- custom_titles[i]
    pdp_data <- partial(model, pred.var = var, train = data)
    pdp_df <- data.frame(pdp_data)
    customize_pdp(pdp_df, var, var_type, title, show_axes)
  })
  
  # Arrange the plots in a grid
  do.call(grid.arrange, c(pdp_plots, ncol = ncol))
}


# Define the list of variables to plot and their types
variables <- c("Video.Duration.Sec.Log", "Colorfulness.Score.SquareRoot", "Semantic.Similarity", 
               "Video.Title.Length","Category.ID", "TN.Human.Faces", "TN.Human.Faces.Emotion", 
               "TN.Human.Faces.Gender", "TN.Logos", "TN.Text", "Video.Title.Emoji", 
               "Video.Title.Sentiment.Vader.Cluster")

var_types <- c("continuous", "continuous", "continuous", 
               "continuous", "categorical", "categorical", 
               "categorical", "categorical", "categorical", "categorical", 
               "categorical", "categorical")

# Define custom titles (placeholders for now)
custom_titles <- c("Video Duration", "Colorfulness", "Semantic Similarity", "Title Length", "Video Category", 
                   "Human Face", "Face Emotion", "Face Gender", "Logos", "Text", "Title Emoji", 
                   "Title Sentiment")

# Call the function with your model, data, variables, and variable types

# View
generate_pdp_plots(rf_model, data_subset, variables, var_types, custom_titles, ncol = 4, title_size = 16, axis_text_size = 14, point_size = 4, show_axes = TRUE)

# Shallow
generate_pdp_plots(rf_model_shallow, data_subset, variables, var_types, custom_titles, ncol = 4, title_size = 16, axis_text_size = 14, point_size = 4, show_axes = TRUE)

# Deep
generate_pdp_plots(rf_model_deep, data_subset, variables, var_types, custom_titles, ncol = 4, title_size = 16, axis_text_size = 14, point_size = 4, show_axes = TRUE)



#### 15 | ALL MODELS - PREDICTIVE VALUES --------------------------------------

calculate_rf_metrics <- function(rf_model, data, actual_values) {
  # Predicted values
  predicted_values <- predict(rf_model, data)$predictions
  
  # Residuals
  residuals <- actual_values - predicted_values
  
  # R-squared
  ss_total <- sum((actual_values - mean(actual_values))^2)
  ss_residual <- sum(residuals^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  # Mean Absolute Error (MAE)
  mae <- mean(abs(residuals))
  
  # Mean Squared Error (MSE)
  mse <- mean(residuals^2)
  
  # Root Mean Squared Error (RMSE)
  rmse <- sqrt(mse)
  
  # Return the metrics as a list
  return(list(
    R_squared = round(r_squared, 3),
    MAE = round(mae, 3),
    MSE = round(mse, 3),
    RMSE = round(rmse, 3)
  ))
}

# Get actual values for the target variable
actual_values <- data_clean$Video.Deep.Engagement.EighthRoot

# Get metrics for the random forest model
metrics_rf <- calculate_rf_metrics(rf_model_deep, data_clean, actual_values)

# Print the metrics
print(metrics_rf)

rf_model
rf_model_shallow
rf_model_deep

