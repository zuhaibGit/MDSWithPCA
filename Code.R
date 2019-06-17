library(grid)

setwd("C:/Users/Ahmed/Desktop/Machine Learning/Assignment 4/cifar-10-binary")
# Read binary file and convert to integer vectors
# [Necessary because reading directly as integer() 
# reads first bit as signed otherwise]
#
# File format is 10000 records following the pattern:
# [label x 1][red x 1024][green x 1024][blue x 1024]
# NOT broken into rows, so need to be careful with "size" and "n"
#
# (See http://www.cs.toronto.edu/~kriz/cifar.html)
labels <- read.table("cifar-10-batches-bin/batches.meta.txt")
images.rgb <- list()
images.lab <- list()
num.images = 10000 # Set to 10000 to retrieve all images per file to memory

# Cycle through all 5 binary files
for (f in 1:6) {
  to.read <- file(paste("cifar-10-batches-bin/data_batch_", f, ".bin", sep=""), "rb")
  for(i in 1:num.images) {
    l <- readBin(to.read, integer(), size=1, n=1, endian="big")
    r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    index <- num.images * (f-1) + i
    images.rgb[[index]] = data.frame(r, g, b)
    images.lab[[index]] = l+1
  }
  close(to.read)  
  remove(l,r,g,b,f,i,index, to.read)
}

#Converts the data frame for each image to a vector of zie 3072
flattened_images <- lapply(images.rgb, function(x) {return(c(as.matrix(x)))})

# function to run sanity check on photos & labels import
drawImage <- function(img) {
  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  #img <- images.rgb[[index]]
  img.r.mat <- matrix(img[,1], ncol=32, byrow = TRUE)
  img.g.mat <- matrix(img[,2], ncol=32, byrow = TRUE)
  img.b.mat <- matrix(img[,3], ncol=32, byrow = TRUE)
  img.col.mat <- rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
  dim(img.col.mat) <- dim(img.r.mat)
  
  # Plot and output label
  grid.raster(img.col.mat, interpolate=FALSE)
  
  # clean up
  remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
}

#Finds the average value for each pixel and returns the resulting
#matrix
find_average_image <- function(lst_of_images) {
  return(Reduce('+', lst_of_images)/length(lst_of_images))
}

#Computes the mean image for each category
mean_images <- list(
mean_images_1 <- find_average_image(images.rgb[which(images.lab == 1)]),
mean_images_2 <- find_average_image(images.rgb[which(images.lab == 2)]),
mean_images_3 <- find_average_image(images.rgb[which(images.lab == 3)]),
mean_images_4 <- find_average_image(images.rgb[which(images.lab == 4)]),
mean_images_5 <- find_average_image(images.rgb[which(images.lab == 5)]),
mean_images_6 <- find_average_image(images.rgb[which(images.lab == 6)]),
mean_images_7 <- find_average_image(images.rgb[which(images.lab == 7)]),
mean_images_8 <- find_average_image(images.rgb[which(images.lab == 8)]),
mean_images_9 <- find_average_image(images.rgb[which(images.lab == 9)]),
mean_images_10 <- find_average_image(images.rgb[which(images.lab == 10)]))

#Flattens the each data frame into a vector of length 3072
flattened_mean_images <- lapply(mean_images, function(x) {return(c(as.matrix(x)))})

#Partitions the images into lists for each class.
images_1 <- flattened_images[which(images.lab == 1)]
images_2 <- flattened_images[which(images.lab == 2)]
images_3 <- flattened_images[which(images.lab == 3)]
images_4 <- flattened_images[which(images.lab == 4)]
images_5 <- flattened_images[which(images.lab == 5)]
images_6 <- flattened_images[which(images.lab == 6)]
images_7 <- flattened_images[which(images.lab == 7)]
images_8 <- flattened_images[which(images.lab == 8)]
images_9 <- flattened_images[which(images.lab == 9)]
images_10 <- flattened_images[which(images.lab == 10)]

#Combines the flattened images of the classes into a data frame
#per class
df_1 <- do.call(rbind, images_1)
df_2 <- do.call(rbind, images_2)
df_3 <- do.call(rbind, images_3)
df_4 <- do.call(rbind, images_4)
df_5 <- do.call(rbind, images_5)
df_6 <- do.call(rbind, images_6)
df_7 <- do.call(rbind, images_7)
df_8 <- do.call(rbind, images_8)
df_9 <- do.call(rbind, images_9)
df_10 <- do.call(rbind, images_10)

pc_1 <- prcomp(df_1, rank = 20)
pc_2 <- prcomp(df_2, rank = 20)
pc_3 <- prcomp(df_3, rank = 20)
pc_4 <- prcomp(df_4, rank = 20)
pc_5 <- prcomp(df_5, rank = 20)
pc_6 <- prcomp(df_6, rank = 20)
pc_7 <- prcomp(df_7, rank = 20)
pc_8 <- prcomp(df_8, rank = 20)
pc_9 <- prcomp(df_9, rank = 20)
pc_10 <- prcomp(df_10, rank = 20)

images_1_reconstructed <- lapply(images_1, function(x) {
    summation <- 0
    for (i in 1:20) {
      summation <- summation + ((t(pc_1$rotation[,i]) %*% (x - flattened_mean_images[[1]])) %*% pc_1$rotation[,i])
    }
    return(summation + flattened_mean_images[[1]])
})
images_2_reconstructed <- lapply(images_2, function(x) {
  summation <- 0
  for (i in 1:20) {
    summation <- summation + ((t(pc_2$rotation[,i]) %*% (x - flattened_mean_images[[2]])) %*% pc_2$rotation[,i])
  }
  return(summation + flattened_mean_images[[2]])
})
images_3_reconstructed <- lapply(images_3, function(x) {
  summation <- 0
  for (i in 1:20) {
    summation <- summation + ((t(pc_3$rotation[,i]) %*% (x - flattened_mean_images[[3]])) %*% pc_3$rotation[,i])
  }
  return(summation + flattened_mean_images[[3]])
})
images_4_reconstructed <- lapply(images_4, function(x) {
  summation <- 0
  for (i in 1:20) {
    summation <- summation + ((t(pc_4$rotation[,i]) %*% (x - flattened_mean_images[[4]])) %*% pc_4$rotation[,i])
  }
  return(summation + flattened_mean_images[[4]])
})
images_5_reconstructed <- lapply(images_5, function(x) {
  summation <- 0
  for (i in 1:20) {
    summation <- summation + ((t(pc_5$rotation[,i]) %*% (x - flattened_mean_images[[5]])) %*% pc_5$rotation[,i])
  }
  return(summation + flattened_mean_images[[5]])
})
images_6_reconstructed <- lapply(images_6, function(x) {
  summation <- 0
  for (i in 1:20) {
    summation <- summation + ((t(pc_6$rotation[,i]) %*% (x - flattened_mean_images[[6]])) %*% pc_6$rotation[,i])
  }
  return(summation + flattened_mean_images[[6]])
})
images_7_reconstructed <- lapply(images_7, function(x) {
  summation <- 0
  for (i in 1:20) {
    summation <- summation + ((t(pc_7$rotation[,i]) %*% (x - flattened_mean_images[[7]])) %*% pc_7$rotation[,i])
  }
  return(summation + flattened_mean_images[[7]])
})
images_8_reconstructed <- lapply(images_8, function(x) {
  summation <- 0
  for (i in 1:20) {
    summation <- summation + ((t(pc_8$rotation[,i]) %*% (x - flattened_mean_images[[8]])) %*% pc_8$rotation[,i])
  }
  return(summation + flattened_mean_images[[8]])
})
images_9_reconstructed <- lapply(images_9, function(x) {
  summation <- 0
  for (i in 1:20) {
    summation <- summation + ((t(pc_9$rotation[,i]) %*% (x - flattened_mean_images[[9]])) %*% pc_9$rotation[,i])
  }
  return(summation + flattened_mean_images[[9]])
})
images_10_reconstructed <- lapply(images_10, function(x) {
  summation <- 0
  for (i in 1:20) {
    summation <- summation + ((t(pc_10$rotation[,i]) %*% (x - flattened_mean_images[[10]])) %*% pc_10$rotation[,i])
  }
  return(summation + flattened_mean_images[[10]])
})

#Finds average mean squared difference between original images and reconstructed images
msd_og_recon_1 <- lapply(c(1:length(images_1)), function(x) {
  return(sum((images_1[[x]] - images_1_reconstructed[[x]])^2))
})
msd_og_recon_2 <- lapply(c(1:length(images_2)), function(x) {
  return(sum((images_2[[x]] - images_2_reconstructed[[x]])^2))
})
msd_og_recon_3 <- lapply(c(1:length(images_3)), function(x) {
  return(sum((images_3[[x]] - images_3_reconstructed[[x]])^2))
})
msd_og_recon_4 <- lapply(c(1:length(images_4)), function(x) {
  return(sum((images_4[[x]] - images_4_reconstructed[[x]])^2))
})
msd_og_recon_5 <- lapply(c(1:length(images_5)), function(x) {
  return(sum((images_5[[x]] - images_5_reconstructed[[x]])^2))
})
msd_og_recon_6 <- lapply(c(1:length(images_6)), function(x) {
  return(sum((images_6[[x]] - images_6_reconstructed[[x]])^2))
})
msd_og_recon_7 <- lapply(c(1:length(images_7)), function(x) {
  return(sum((images_7[[x]] - images_7_reconstructed[[x]])^2))
})
msd_og_recon_8 <- lapply(c(1:length(images_8)), function(x) {
  return(sum((images_8[[x]] - images_8_reconstructed[[x]])^2))
})
msd_og_recon_9 <- lapply(c(1:length(images_9)), function(x) {
  return(sum((images_9[[x]] - images_9_reconstructed[[x]])^2))
})
msd_og_recon_10 <- lapply(c(1:length(images_10)), function(x) {
  return(sum((images_10[[x]] - images_10_reconstructed[[x]])^2))
})

#Average MSE. To be plotted.

average_error_per_class <- c(2620501,3950676,2447698,3116479,2180391,3231113,2630244,3441091,
                             2440635,4021094)
class_labels <- c("airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck")
barplot(average_error_per_class, names.arg = class_labels, xlab="Classes", ylab="Average Error per Class")

#Computes distance matrix between mean images for each class
dist_mat <- dist(do.call(rbind, flattened_mean_images), diag = T, upper = T)
dist_mat <- (as.matrix(dist_mat))^2
setwd("C:/Users/Ahmed/Desktop/Machine Learning/Assignment 4")
write.csv(dist_mat, "partb_distances.csv")
#Performs MDS on MSD from standard distance matrix
mat_A <- diag(1,10) - (1/10)*matrix(1,10,10)
mat_W <- (-1/2)*mat_A%*%dist_mat%*%t(mat_A)
ev_W <- eigen(mat_W)
new_points <- ev_W$vectors[,c(1:2)] %*% (diag(ev_W$values)[c(1,2),c(1,2)])^(0.5)
plot(new_points[,1], new_points[,2])
text(new_points[,1], new_points[,2], labels = class_labels, cex=0.7, pos = 1)

#Finds distances between two image classes as described in part C of Assignment
find_distance <- function(num1, num2) {
  recon_1_2 <- lapply(eval(parse(text=paste0("images_",num1))), function(x) {
    summation <- 0
    pc <- eval(parse(text=paste0("pc_",num2)))
    for (i in 1:20) {
      summation <- summation + ((t(pc$rotation[,i]) %*% (x - flattened_mean_images[[num1]])) %*% pc$rotation[,i])
    }
    return(summation + flattened_mean_images[[num1]])
  })
  msd_recon_1_2 <- lapply(c(1:6000), function(x) {
    return(sum((eval(parse(text=paste0("images_",num1)))[[x]] - recon_1_2[[x]])^2))
  })
  
  recon_2_1 <- lapply(eval(parse(text=paste0("images_",num2))), function(x) {
    summation <- 0
    pc <- eval(parse(text=paste0("pc_",num1)))
    for (i in 1:20) {
      summation <- summation + ((t(pc$rotation[,i]) %*% (x - flattened_mean_images[[num2]])) %*% pc$rotation[,i])
    }
    return(summation + flattened_mean_images[[num2]])
  })
  msd_recon_2_1 <- lapply(c(1:6000), function(x) {
    return(sum((eval(parse(text=paste0("images_",num2)))[[x]] - recon_2_1[[x]])^2))
  })  
  
  return((mean(unlist(msd_recon_1_2)) + mean(unlist(msd_recon_2_1)))/2)
}


#Computes distance matrix between mean images for each class
E11 <- find_distance(1,1)
E12 <- find_distance(1,2)
E13 <- find_distance(1,3)
E14 <- find_distance(1,4)
E15 <- find_distance(1,5)
E16 <- find_distance(1,6)
E17 <- find_distance(1,7)
E18 <- find_distance(1,8)
E19 <- find_distance(1,9)
E110 <- find_distance(1,10)
E22 <- find_distance(2,2)
E23 <- find_distance(2,3)
E24 <- find_distance(2,4)
E25 <- find_distance(2,5)
E26 <- find_distance(2,6)
E27 <- find_distance(2,7)
E28 <- find_distance(2,8)
E29 <- find_distance(2,9)
E210 <- find_distance(2,10)
E33 <- find_distance(3,3)
E34 <- find_distance(3,4)
E35 <- find_distance(3,5)
E36 <- find_distance(3,6)
E37 <- find_distance(3,7)
E38 <- find_distance(3,8)
E39 <- find_distance(3,9)
E310 <- find_distance(3,10)
E44 <- find_distance(4,4)
E45 <- find_distance(4,5)
E46 <- find_distance(4,6)
E47 <- find_distance(4,7)
E48 <- find_distance(4,8)
E49 <- find_distance(4,9)
E410 <- find_distance(4,10)
E55 <- find_distance(5,5)
E56 <- find_distance(5,6)
E57 <- find_distance(5,7)
E58 <- find_distance(5,8)
E59 <- find_distance(5,9)
E510 <- find_distance(5,10)
E66 <- find_distance(6,6)
E67 <- find_distance(6,7)
E68 <- find_distance(6,8)
E69 <- find_distance(6,9)
E610 <- find_distance(6,10)
E77 <- find_distance(7,7)
E78 <- find_distance(7,8)
E79 <- find_distance(7,9)
E710 <- find_distance(7,10)
E88 <- find_distance(8,8)
E89 <- find_distance(8,9)
E810 <- find_distance(8,10)
E99 <- find_distance(9,9)
E910 <- find_distance(9,10)
E1010 <- find_distance(10,10)

dist_mat_2 <- matrix(c(E11,E12,E13,E14,E15,E16,E17,E18,E19,E110,
                       E12,E22,E23,E24,E25,E26,E27,E28,E29,E210,
                       E13,E23,E33,E34,E35,E36,E37,E38,E39,E310,
                       E14,E24,E34,E44,E45,E46,E47,E48,E49,E410,
                       E15,E25,E35,E45,E55,E56,E57,E58,E59,E510,
                       E16,E26,E36,E46,E56,E66,E67,E68,E69,E610,
                       E17,E27,E37,E47,E57,E67,E77,E78,E79,E710,
                       E18,E28,E38,E48,E58,E68,E78,E88,E89,E810,
                       E19,E29,E39,E49,E59,E69,E79,E89,E99,E910,
                       E110,E210,E310,E410,E510,E610,E710,E810,E910,E1010),
                       10,10)

#Performs MDS on MSD from standard distance matrix
mat_A_2 <- diag(1,10) - (1/10)*matrix(1,10,10)
mat_W_2 <- (-1/2)*mat_A_2%*%dist_mat_2%*%t(mat_A_2)
ev_W_2 <- eigen(mat_W_2)
new_points_2 <- ev_W_2$vectors[,c(1:2)] %*% (diag(ev_W_2$values)[c(1,2),c(1,2)])^(0.5)
plot(new_points_2[,1], new_points_2[,2])
text(new_points_2[,1], new_points_2[,2], labels = class_labels, cex=0.7, pos = 1)