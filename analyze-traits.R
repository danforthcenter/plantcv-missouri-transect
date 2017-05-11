library(ggplot2)
library(lubridate)
library(gtools)
library(scales)
library(gplots)
library(nlme)
library(mvtnorm)
library(dplyr)
library(grid)
library(reshape2)

########################################################################################
# Analyze VIS data
########################################################################################
############################################
# Zoom correction
############################################
zoom.lm = lm(zoom.camera ~ zoom, data=data.frame(zoom=c(1,6000), zoom.camera=c(1,6)))

# Download data for a reference object imaged at different zoom levels
if (!file.exists('zoom_calibration_data.txt')) {
  download.file('http://files.figshare.com/2084101/zoom_calibration_data.txt',
                'zoom_calibration_data.txt')
}

# Read zoom calibrartion data
z.data = read.table(file="zoom_calibration_data.txt", sep="\t", header=TRUE)

# Calculate px per cm
z.data$px_cm = z.data$length_px / z.data$length_cm

# Convert LemnaTec zoom units to camera zoom units
z.data$zoom.camera = predict(object = zoom.lm, newdata=z.data)

# Zoom correction for area
area.coef = coef(nls(log(rel_area) ~ log(a * exp(b * zoom.camera)),
                     z.data, start = c(a = 1, b = 0.01)))
area.coef = data.frame(a=area.coef[1], b=area.coef[2])
area.nls = nls(rel_area ~ a * exp(b * zoom.camera),
               data = z.data, start=c(a=area.coef$a, b=area.coef$b))

# Zoom correction for length
len.poly = lm(px_cm ~ zoom.camera + I(zoom.camera^2),
              data=z.data[z.data$camera == 'VIS SV',])

############################################
# Read data and format for analysis
############################################
# Planting date
planting_date = as.POSIXct("2015-08-10")

# Read VIS data
vis.data = read.table(file="plantcv_results.csv", sep=',', header=TRUE)
# Read metadata
metadata = read.table(file="../sample_data/TM007_E_082415_barcodes.csv", sep=",", header=TRUE)

# Use barcodes to assign genotype and treatment labels
vis.data = merge(vis.data, metadata, by.x = "plantbarcode", by.y = "Barcodes")

# Convert timestamp from text to date-time
vis.data$timestamp = ymd_hms(vis.data$timestamp)

# Days after planting
vis.data$dap = as.numeric(vis.data$timestamp - planting_date)

# Integer day
vis.data$day = as.integer(vis.data$dap)

# Convert LemnaTec zoom units to camera zoom units
vis.data$zoom = vis.data$tv0_zoom
vis.data$zoom = as.integer(gsub('z', '', vis.data$zoom))
vis.data$zoom.camera = predict(object = zoom.lm, newdata = vis.data)
vis.data$rel_area = predict(object = area.nls, newdata = vis.data)
vis.data$px_cm = predict(object = len.poly, newdata = vis.data)

############################################
# Build traits table
############################################
traits = data.frame(plantbarcode = vis.data$plantbarcode, timestamp = vis.data$timestamp,
                    genotype = vis.data$Genotype, dap = vis.data$dap, day = vis.data$day)

# Zoom correct TV and SV area
traits$tv_area = vis.data$tv0_area / vis.data$rel_area
traits$sv_area = (vis.data$sv0_area / vis.data$rel_area) +
  (vis.data$sv90_area / vis.data$rel_area)
traits$area = traits$tv_area + traits$sv_area

# Zoom correct height
traits$height = ((vis.data$sv0_height_above_bound / vis.data$px_cm) +
                   (vis.data$sv90_height_above_bound / vis.data$px_cm)) / 2

# Remove dead/slow-growing plants
max_area = summarise(group_by(traits, plantbarcode), max(area))
dead.plants = max_area[max_area$`max(area)` < 100000,]$plantbarcode
dead.plants = droplevels(dead.plants)
traits = traits[!traits$plantbarcode %in% dead.plants,]
traits$plantbarcode = droplevels(traits$plantbarcode)

############################################
# Leaf area analysis
############################################
leaf.plot = ggplot(traits, aes(x=dap, y=area)) + 
                   geom_line(aes(group=plantbarcode)) + 
                   geom_smooth(method = "loess") +
                   scale_x_continuous("Days after planting") +
                   scale_y_continuous("Projected leaf area (px)") +
                   theme_bw()
ggsave(filename = "plant_leaf_area.png", plot = leaf.plot)

############################################
# Height analysis
############################################
height.plot = ggplot(traits, aes(x=dap, y=height)) + 
                     geom_line(aes(group=plantbarcode)) + 
                     geom_smooth(method = "loess") +
                     scale_x_continuous("Days after planting") +
                     scale_y_continuous("Height (cm)") +
                     theme_bw()
ggsave(filename = "plant_height.png", plot = height.plot)
