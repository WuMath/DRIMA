library(Seurat)
# library(clustree)
library(cowplot)
library(data.table)
library(ggplot2)
library(patchwork)
library(stringr)
# library(qs)
library(Matrix)


data_root <- "/home/wuyan/dygmamba_project/NewRealPlan/Case1/process/"

traj <- "myeloid"

data_path <- paste0(data_root, traj, "/process/")

# The only need input are the atac_dir and rna_dir

atac_dir <- paste0(data_path,  "R/atac/")

rna_dir <- paste0(data_path,  "R/rna/")

############################################################################################
#******************************** Data Read --- ATAC ***************************************
############################################################################################

mtx <- readMM(paste0(atac_dir, "sparse.mtx"))

cellinfo <- fread(paste0(atac_dir, "cellinfo.csv"), header = T, data.table = F)

atacinfo <- fread(paste0(atac_dir, "atacinfo.csv"), header = T, data.table = F)

rownames(mtx) <- cellinfo$V1
colnames(mtx) <- atacinfo$V1

meta <- cellinfo
rownames(meta) <- cellinfo$V1

if(identical(rownames(meta), rownames(mtx))){
        # 行需为 feature
    print("Info identical, Can create Seurat Object")
    mtx_transposed <- t(mtx) 

    atac_seurat <- CreateSeuratObject(count = mtx_transposed, meta.data = meta, assay = "ATAC")
}

############################################################################################
#******************************** Data Read --- RNA ***************************************
############################################################################################


rna_mtx <- readMM(paste0(rna_dir, "sparse.mtx"))

rna_cellinfo <- fread(paste0(rna_dir, "cellinfo.csv"), header = T, data.table = F)

rna_info <- fread(paste0(rna_dir, "rnainfo.csv"), header = T, data.table = F)

rownames(rna_mtx) <- rna_cellinfo$V1
colnames(rna_mtx) <- rna_info$Geneid

meta <- rna_cellinfo
rownames(meta) <- rna_cellinfo$V1
 
print("Create Seurat Object")
rna_mtx_transposed <- t(rna_mtx)

rna_seurat <- CreateSeuratObject(count = rna_mtx_transposed, meta.data = meta, assay = "RNA")


umap_csv <- read.csv(paste0(data_path, "umap_coords.csv"), row.names = 1)

# 对齐细胞顺序
common_cells <- intersect(rownames(umap_csv), colnames(rna_seurat))
umap_matrix <- as.matrix(umap_csv[common_cells, c("UMAP_1", "UMAP_2")])
colnames(umap_matrix) <- c("umap_1", "umap_2")   # Seurat/Slingshot 需要小写

# 注入到 Seurat 对象
rna_seurat <- rna_seurat[, common_cells]
rna_seurat[["umap"]] <- CreateDimReducObject(
    embeddings = umap_matrix,
    key = "umap_",
    assay = DefaultAssay(rna_seurat)
)

cat("已导入 Python UMAP:", nrow(umap_matrix), "cells\n")

############################################################################################
#******************************** Data Preprocess   ***************************************
############################################################################################

library(tidyverse)
library(slingshot)
library(Seurat)
library(S4Vectors)
library(SingleCellExperiment)
library(RColorBrewer)
library(arrow)

# 数据预处理
rna_seurat <- NormalizeData(rna_seurat)
rna_seurat <- FindVariableFeatures(rna_seurat)
rna_seurat <- ScaleData(rna_seurat)
rna_seurat <- RunPCA(rna_seurat)

# 提取 PCA 坐标（细胞 x PC）
pca_coords <- rna_seurat@reductions$pca@cell.embeddings[, 1:30]

# 找出重复的行（坐标）的索引
duplicate_cells <- duplicated(pca_coords)
num_duplicates <- sum(duplicate_cells)
cat(paste("找到", num_duplicates, "个重复的细胞 (在PCA空间中).\n"))

duplicate_cell_names <- rownames(pca_coords)[duplicate_cells]
valid_cells <- setdiff(colnames(rna_seurat), duplicate_cell_names)
original_seurat <- rna_seurat
rna_seurat <- subset(rna_seurat, cells = valid_cells)

# 细胞聚类
rna_seurat <- FindNeighbors(rna_seurat, dims = 1:30)
rna_seurat <- FindClusters(rna_seurat, resolution = 0.5)

# 非线性降维（UMAP/t-SNE）
# rna_seurat <- RunUMAP(rna_seurat, dims = 1:30)
# rna_seurat <- RunTSNE(rna_seurat, dims = 1:30)


############################################################################################
#******************************** Trajectory inference *************************************
############################################################################################

# 方法A：直接使用聚类结果
sce <- as.SingleCellExperiment(rna_seurat)

# 运行slingshot - 使用UMAP坐标
sce <- slingshot(sce, 
                clusterLabels = rna_seurat@meta.data$cell_type, 
                reducedDim = "UMAP",
                start.clus = "HSC")  # 根据生物学知识指定起始簇

# 提取轨迹信息
trajectories <- slingshot::SlingshotDataSet(sce)

# 查看轨迹信息
summary(trajectories)

pseudotime <- slingshot::slingPseudotime(sce)
curves <- slingshot::slingCurves(sce)

# 查看每条 lineage 的细胞数
cells_per_lineage <- apply(pseudotime, 2, function(x) sum(!is.na(x)))
print("Number of cells in each lineage:")
print(cells_per_lineage)

# ========== 修改部分：取所有 lineage 的均值伪时间 ==========
avg_pseudotime <- rowMeans(pseudotime, na.rm = TRUE)

# 移除全部 lineage 都是 NA 的细胞（极少见）
valid_cells <- !is.nan(avg_pseudotime)
avg_pseudotime_clean <- avg_pseudotime[valid_cells]
cell_barcodes <- names(avg_pseudotime_clean)

print(paste("Total cells:", nrow(pseudotime)))
print(paste("Valid cells after averaging:", length(avg_pseudotime_clean)))

# 创建数据框
pseudotime_df <- data.frame(
  cell_barcode = cell_barcodes,
  pseudotime = avg_pseudotime_clean,
  stringsAsFactors = FALSE
)

pseudotime_df <- pseudotime_df[order(pseudotime_df$pseudotime), ]

pseudotime_df_unique <- pseudotime_df[!duplicated(pseudotime_df$pseudotime), ]

print("Before filter")
print(dim(pseudotime_df))
print("After filter")
print(dim(pseudotime_df_unique))


############################################################################################
#********************************      Data Save       *************************************
############################################################################################

# 定义输出文件名
output_filename <- paste0(data_path, "avg_lineage_pseudotime.csv")

# 保存为CSV
write.csv(pseudotime_df_unique, 
          file = output_filename, 
          row.names = FALSE, 
          quote = FALSE)

print(paste("Data saved to:", output_filename))