library(Seurat)
# library(clustree)
library(cowplot)
library(data.table)
library(ggplot2)
library(patchwork)
library(stringr)
# library(qs)
library(Matrix)


cell_type <- "Panc1"

data_path <- paste0("/home/liyang/BioWuYan/dygmamba_project/data/cell_line/", cell_type, "/process/")


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

# 统计有多少重复的细胞
num_duplicates <- sum(duplicate_cells)
cat(paste("找到", num_duplicates, "个重复的细胞 (在PCA空间中).\n"))

# 获取重复细胞的名称
duplicate_cell_names <- rownames(pca_coords)[duplicate_cells]

# 过滤掉重复的细胞名称
valid_cells <- setdiff(colnames(rna_seurat), duplicate_cell_names)

original_seurat <- rna_seurat

# 创建一个没有重复细胞的 Seurat 子集对象
rna_seurat <- subset(rna_seurat, cells = valid_cells)

# 细胞聚类
rna_seurat <- FindNeighbors(rna_seurat, dims = 1:30)
rna_seurat <- FindClusters(rna_seurat, resolution = 0.5)

# 非线性降维（UMAP/t-SNE）
rna_seurat <- RunUMAP(rna_seurat, dims = 1:30)
rna_seurat <- RunTSNE(rna_seurat, dims = 1:30)


############################################################################################
#******************************** Trajectory inference *************************************
############################################################################################

# 方法A：直接使用聚类结果
sce <- as.SingleCellExperiment(rna_seurat)

# 运行slingshot - 使用UMAP坐标
sce <- slingshot(sce, 
                clusterLabels = "seurat_clusters", 
                reducedDim = "UMAP",
                start.clus = "0")  # 根据生物学知识指定起始簇

# 提取轨迹信息
trajectories <- slingshot::SlingshotDataSet(sce)

# 查看轨迹信息
summary(trajectories)

# 获取伪时间值
pseudotime <- slingshot::slingPseudotime(sce)
curves <- slingshot::slingCurves(sce)

# 计算每个轨迹中的细胞数量（非NA值的数量）
cells_per_lineage <- apply(pseudotime, 2, function(x) sum(!is.na(x)))

print("Number of cells in each lineage:")
print(cells_per_lineage)

# 找到细胞数量最多的轨迹
max_cells_lineage <- names(which.max(cells_per_lineage))
print(paste("Lineage with most cells:", max_cells_lineage))

# 提取该轨迹的伪时间值
max_lineage_pseudotime <- pseudotime[, max_cells_lineage]

# 移除NA值（不属于该轨迹的细胞）
valid_cells <- !is.na(max_lineage_pseudotime)
max_lineage_pseudotime_clean <- max_lineage_pseudotime[valid_cells]


# 获取细胞名称（barcodes）
cell_barcodes <- rownames(pseudotime)[valid_cells]

# 创建包含细胞信息和伪时间的数据框
max_lineage_df <- data.frame(
  cell_barcode = cell_barcodes,
  pseudotime = max_lineage_pseudotime_clean,
  lineage = max_cells_lineage,
  stringsAsFactors = FALSE
)

# 按伪时间排序
max_lineage_df <- max_lineage_df[order(max_lineage_df$pseudotime), ]

# 查看前几行
print("Top cells in the lineage:")
print(max_lineage_df)


max_lineage_df_unique <- max_lineage_df[!duplicated(max_lineage_df$pseudotime), ]

print("Before filter")
print(dim(max_lineage_df))

print("After filter")
print(dim(max_lineage_df_unique))

############################################################################################
#********************************      Data Save       *************************************
############################################################################################

# 定义输出文件名
output_filename <- paste0(data_path, "max_cells_lineage_", max_cells_lineage, "_pseudotime.csv")

# 保存为CSV
write.csv(max_lineage_df_unique, 
          file = output_filename, 
          row.names = FALSE, 
          quote = FALSE)

print(paste("Data saved to:", output_filename))