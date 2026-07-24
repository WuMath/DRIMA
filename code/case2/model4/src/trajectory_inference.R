# ============================================================
# trajectory_inference.R — AD 数据版本
# 适用于: 单细胞类型（全是 Microglia 或 Astrocyte）
# 用 Leiden 子聚类作为 clusterLabels，月龄最小的 cluster 作为 start
# ============================================================

library(Seurat)
library(slingshot)
library(data.table)
library(Matrix)

# ---- 路径设置 ----

data_path <- paste0("/home/wuyan/dygmamba_project/NewRealPlan/case2/data/AD/process/",
                    "model4_WT_Astrocyte", "/process/")

cat("数据路径:", data_path, "\n")

# ---- 读取数据 ----
rna_dir <- paste0(data_path, "R/rna/")
mtx <- readMM(paste0(rna_dir, "sparse.mtx"))
cellinfo <- fread(paste0(rna_dir, "cellinfo.csv"), header = TRUE, data.table = FALSE)
rnainfo  <- fread(paste0(rna_dir, "rnainfo.csv"), header = TRUE, data.table = FALSE)

rownames(mtx) <- cellinfo$V1
colnames(mtx) <- rnainfo$V1

meta <- cellinfo
rownames(meta) <- cellinfo$V1

rna_seurat <- CreateSeuratObject(counts = t(mtx), meta.data = meta, assay = "RNA")

cat("细胞数:", ncol(rna_seurat), "\n")

# ---- 标准处理 ----
rna_seurat <- NormalizeData(rna_seurat)
rna_seurat <- FindVariableFeatures(rna_seurat, nfeatures = 2000)
rna_seurat <- ScaleData(rna_seurat)
rna_seurat <- RunPCA(rna_seurat, npcs = 20)

# ---- 导入 Python 的 UMAP ----
umap_file <- paste0(data_path, "umap_coords.csv")
if (file.exists(umap_file)) {
    umap_csv <- read.csv(umap_file, row.names = 1)
    common <- intersect(rownames(umap_csv), colnames(rna_seurat))
    umap_mat <- as.matrix(umap_csv[common, c("UMAP_1", "UMAP_2")])
    colnames(umap_mat) <- c("umap_1", "umap_2")
    rna_seurat <- rna_seurat[, common]
    rna_seurat[["umap"]] <- CreateDimReducObject(
        embeddings = umap_mat, key = "umap_", assay = "RNA")
    cat("已导入 Python UMAP:", nrow(umap_mat), "cells\n")
} else {
    cat("未找到 umap_coords.csv，自行计算 UMAP\n")
    rna_seurat <- FindNeighbors(rna_seurat, dims = 1:15)
    rna_seurat <- RunUMAP(rna_seurat, dims = 1:15)
}

# ============================================================
# 关键步骤: Leiden 子聚类 + 确定起始 cluster
# ============================================================
rna_seurat <- FindNeighbors(rna_seurat, dims = 1:15)
rna_seurat <- FindClusters(rna_seurat, resolution = 0.3)
# resolution=0.3 产生较少的 cluster（3-6个），适合 Slingshot

cat("\nLeiden 子聚类结果:\n")
print(table(rna_seurat@meta.data$seurat_clusters))

# ---- 找到月龄最小（最早期）的 cluster 作为起点 ----
# 计算每个 cluster 的平均月龄
cluster_mean_age <- tapply(
    as.numeric(rna_seurat@meta.data$age),
    rna_seurat@meta.data$seurat_clusters,
    mean
)
cat("\n每个 cluster 的平均月龄:\n")
print(round(cluster_mean_age, 2))

# 平均月龄最小的 cluster = 起点
start_cluster <- names(which.min(cluster_mean_age))
cat("\n起始 cluster:", start_cluster, 
    "(平均月龄 =", round(cluster_mean_age[start_cluster], 1), "月)\n")

# ---- 验证: 每个 cluster 的月龄组成 ----
cat("\nCluster × Age 交叉表:\n")
print(table(rna_seurat@meta.data$seurat_clusters, rna_seurat@meta.data$age))

# ============================================================
# Slingshot 轨迹推断
# ============================================================
sce <- as.SingleCellExperiment(rna_seurat)

sce <- slingshot(sce,
                 clusterLabels = rna_seurat@meta.data$seurat_clusters,
                 reducedDim = "UMAP",
                 start.clus = start_cluster)

cat("\n推断到", ncol(slingPseudotime(sce)), "条 lineage\n")

# ---- 提取伪时序 ----
pt <- slingPseudotime(sce)

# 如果有多条 lineage，取平均（或取最长的）
if (ncol(pt) == 1) {
    pseudotime_vals <- pt[, 1]
} else {
    # 方案 A: 取每个细胞所在 lineage 中 non-NA 的平均
    pseudotime_vals <- rowMeans(pt, na.rm = TRUE)
    # 方案 B: 取最长的那条（包含最多非 NA 细胞的）
    # longest <- which.max(colSums(!is.na(pt)))
    # pseudotime_vals <- pt[, longest]
}

# 去掉 NA
valid <- !is.na(pseudotime_vals)
cat("有效伪时序细胞:", sum(valid), "/", length(valid), "\n")

# # 归一化到 [0, 1]
# pt_min <- min(pseudotime_vals[valid])
# pt_max <- max(pseudotime_vals[valid])
# pseudotime_norm <- (pseudotime_vals - pt_min) / (pt_max - pt_min)

# # ---- 验证伪时序方向 ----
# cat("\n伪时序方向验证（应随月龄递增）:\n")
# ages <- as.numeric(rna_seurat@meta.data$age[valid])
# for (a in sort(unique(ages))) {
#     mask <- ages == a
#     cat(sprintf("  %5.1f 月: mean_pt = %.3f, n = %d\n",
#                 a, mean(pseudotime_norm[valid][mask]), sum(mask)))
# }

# # 如果方向反了（老的伪时序小），翻转
# age_pt_cor <- cor(ages, pseudotime_norm[valid])
# cat("\n月龄-伪时序相关性:", round(age_pt_cor, 3), "\n")
# if (age_pt_cor < 0) {
#     cat("  [翻转] 伪时序方向与月龄相反，执行翻转\n")
#     pseudotime_norm <- 1 - pseudotime_norm
# }

# # ---- 保存 ----
# result <- data.frame(
#     cell_barcode = colnames(rna_seurat)[valid],
#     pseudotime = pseudotime_norm[valid]
# )

# write.csv(result,
#           paste0(data_path, "max_cells_lineage_Lineage1_pseudotime.csv"),
#           row.names = FALSE)

# # 兼容格式
# write.csv(result,
#           paste0(data_path, "avg_lineage_pseudotime.csv"),
#           row.names = FALSE)

# cat("\n伪时序已保存:", nrow(result), "cells\n")
# cat("完成！\n")

pseudotime_raw <- pseudotime_vals

# ---- 验证伪时序方向 ----
cat("\n伪时序方向验证（应随月龄递增）:\n")
ages <- as.numeric(rna_seurat@meta.data$age[valid])
for (a in sort(unique(ages))) {
    mask <- ages == a
    cat(sprintf("  %5.1f 月: mean_pt = %.3f, n = %d\n",
                a, mean(pseudotime_raw[valid][mask]), sum(mask)))
}

# 如果方向反了，翻转（用 max - val 而不是 1 - val）
age_pt_cor <- cor(ages, pseudotime_raw[valid])
cat("\n月龄-伪时序相关性:", round(age_pt_cor, 3), "\n")
if (age_pt_cor < 0) {
    cat("  [翻转] 伪时序方向与月龄相反，执行翻转\n")
    pt_max <- max(pseudotime_raw[valid])
    pseudotime_raw[valid] <- pt_max - pseudotime_raw[valid]
}

# ---- 保存 ----
result <- data.frame(
    cell_barcode = colnames(rna_seurat)[valid],
    pseudotime   = pseudotime_raw[valid]   # 原始 Slingshot 伪时序，无归一化
)

write.csv(result,
          paste0(data_path, "max_cells_lineage_Lineage1_pseudotime.csv"),
          row.names = FALSE)
write.csv(result,
          paste0(data_path, "avg_lineage_pseudotime.csv"),
          row.names = FALSE)

cat("\n伪时序范围:", round(min(result$pseudotime), 2),
    "~", round(max(result$pseudotime), 2), "\n")
cat("伪时序已保存:", nrow(result), "cells\n")
cat("完成！\n")