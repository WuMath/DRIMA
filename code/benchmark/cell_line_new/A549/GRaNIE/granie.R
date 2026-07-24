library(readr)
library(GRaNIE)
library(Seurat)
library(Signac)
library(cowplot)
library(data.table)
library(ggplot2)
library(patchwork)
library(stringr)
library(Matrix)
library(org.Hs.eg.db) # 人类数据
library(AnnotationDbi)
library(dplyr)
library(tibble)


cell_type <- "A549"

data_path <- paste0("/home/wuyan/dygmamba_project/data/cell_line/", cell_type, "/process/")

output_path <- paste0("/home/wuyan/dygmamba_project/data/cell_line/", cell_type, "/GRaNIE_output/")

dir.create(output_path, recursive = TRUE, showWarnings = FALSE)

# The only need input are the atac_dir and rna_dir

atac_dir <- paste0(data_path,  "R/atac/")

rna_dir <- paste0(data_path,  "R/rna/")

################################################
#*************** Data Read --- ATAC ************
################################################

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

rownames(atac_seurat) <- sub("-", ":", rownames(atac_seurat)) # 确保只有第一个是冒号

######################################################
#****************** Data Read --- RNA ****************
######################################################


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

######################################################
#****************** Data Process ****************
######################################################


n_bins <- 50 

rna_seurat <- NormalizeData(rna_seurat)
rna_seurat <- FindVariableFeatures(rna_seurat)
rna_seurat <- ScaleData(rna_seurat)
rna_seurat <- RunPCA(rna_seurat, verbose = FALSE) # 生成 PCA 降维

rna_seurat <- RunUMAP(rna_seurat, dims = 1:30)
rna_kmeans <- kmeans(Embeddings(rna_seurat, "umap"), centers = n_bins)

rna_seurat$metacell_id <- paste0("bin_", rna_kmeans$cluster)


atac_seurat <- RunTFIDF(atac_seurat)
atac_seurat <- FindTopFeatures(atac_seurat, min.cutoff = 'q0')
atac_seurat <- RunSVD(atac_seurat) # ATAC 的降维通常叫 LSI 或 SVD

# 3. 运行 UMAP (基于 lsi)
atac_seurat <- RunUMAP(atac_seurat, reduction = 'lsi', dims = 2:30)

atac_kmeans <- kmeans(Embeddings(atac_seurat, "umap"), centers = n_bins)
atac_seurat$metacell_id <- paste0("bin_", atac_kmeans$cluster)



# --- 处理 RNA ---
DefaultAssay(rna_seurat) <- "RNA" # 显式设置
# 聚合
rna_assay_name <- DefaultAssay(rna_seurat)
pb_list_rna <- AggregateExpression(rna_seurat, group.by = "metacell_id", 
                                   assays = rna_assay_name, slot = "counts")
pb_rna <- pb_list_rna[[rna_assay_name]] # 动态提取

# --- 处理 ATAC ---
atac_assay_name <- DefaultAssay(atac_seurat)
pb_list_atac <- AggregateExpression(atac_seurat, group.by = "metacell_id", 
                                    assays = atac_assay_name, slot = "counts")
pb_atac <- pb_list_atac[[atac_assay_name]]


# --- 对齐数据 ---
common_bins <- intersect(colnames(pb_rna), colnames(pb_atac))
pb_rna <- pb_rna[, common_bins]
pb_atac <- pb_atac[, common_bins]


######################################################
#****************** GRN inference ****************
######################################################


# 1. 修正 ATAC 矩阵：将行名提取为 "peakID" 列
pb_atac_df <- as.data.frame(pb_atac) %>% 
  rownames_to_column(var = "peakID")

# 2. 修正 RNA 矩阵：将行名提取为 "geneID" 列
pb_rna_df <- as.data.frame(pb_rna) %>% 
  rownames_to_column(var = "geneID")

# 检查一下现在长什么样 (你应该能看到第一列变成了 ID)
print(head(pb_atac_df[, 1:5])) 

metadata_df <- data.frame(sample_id = common_bins, cell_type = "SingleType")

GRN <- initializeGRN(objectMetadata = list(name = "Metacell_GRN"), 
                      genomeAssembly = "hg38", 
                      outputFolder = paste0(output_path, "GRaNIE_Results"))



current_symbols <- pb_rna_df$geneID # 假设你之前的 ID 列叫 geneID
ensembl_ids <- mapIds(org.Hs.eg.db,
                      keys = current_symbols,
                      column = "ENSEMBL",
                      keytype = "SYMBOL",
                      multiVals = "first")


pb_rna_df_fixed <- pb_rna_df %>%
            mutate(ensemblID = ensembl_ids[geneID]) %>% # 根据旧 ID 匹配新 ID
            filter(!is.na(ensemblID)) %>%               # 去掉没找到 Ensembl ID 的基因
            distinct(ensemblID, .keep_all = TRUE)       # 去掉重复的 ID (非常重要！)

pb_rna_clean <- pb_rna_df_fixed %>%
            dplyr::select(ensemblID, where(is.numeric)) 


GRN <- addData(GRN, 
               counts_peaks = pb_atac_df,      # ATAC 数据不变
               counts_rna = pb_rna_clean,   # 使用修复后的 RNA 数据
               sampleMetadata = metadata_df,   # 元数据不变
               idColumn_peaks = "peakID",      # ATAC 的 ID 列名
               idColumn_RNA = "ensemblID",     # ⚠️ 注意：这里改成新的列名
               force = TRUE)


# 1. 从本地数据库提取 Symbol 到 Ensembl 的对照表
keys <- keys(org.Hs.eg.db, keytype = "SYMBOL")
mapping <- select(org.Hs.eg.db, 
                  keys = keys, 
                  columns = c("ENSEMBL", "SYMBOL"), 
                  keytype = "SYMBOL")

# 2. 整理成 GRaNIE 格式 (使用了 dplyr:: 前缀来防止报错)
translation_table_local <- mapping %>%
  dplyr::rename(TF_name = SYMBOL, TF_ensembl = ENSEMBL) %>%
  dplyr::filter(!is.na(TF_ensembl)) %>%        # <--- 修正点：加上 dplyr::
  dplyr::distinct(TF_name, .keep_all = TRUE)   # <--- 修正点：加上 dplyr::

print(paste("本地构建了", nrow(translation_table_local), "个基因的对照表"))

# 使用本地表运行 addTFBS
GRN <- addTFBS(GRN, 
               source = "JASPAR2024", 
               translationTable = translation_table_local)



# 2. 计算 TFBS 与 Peak 的重叠
# 这一步现在会使用 JASPAR 的 Motif 进行扫描
GRN <- overlapPeaksAndTFBS(GRN)


# 3. 计算 TF-Peak 链接
GRN <- addConnections_TF_peak(GRN, 
                              corMethod = "pearson")

# 4. 计算 Peak-Gene 链接
GRN <- addConnections_peak_gene(GRN,  
                                corMethod = "pearson",
                                promoterRange = 250000,
                                forceRerun = TRUE)
# 5. 过滤并生成网络
# 对于 Metacell 数据，建议先用 0.2 的 FDR 试试水，如果结果太多再收紧到 0.1
GRN <- filterGRNAndConnectGenes(GRN, TF_peak.fdr.threshold = 1,
                                peak_gene.fdr.threshold = 1,
                                peak_gene.r_range       = c(-1, 1))


# 6. 导出结果
results_df <- getGRNConnections(GRN, type = "all.filtered")

write.csv(results_df, paste0(output_path, "GRaNIE_Full_Links.csv"), row.names = FALSE)
# 打印结果概览
print(paste("找到的连接数:", nrow(results_df)))


tf_region_df <- results_df %>%
  dplyr::select(TF.name, TF.ENSEMBL, peak.ID, TF_peak.r, TF_peak.fdr) %>%
  dplyr::distinct() # 去重，因为一个 TF-Peak 组合可能对应多个靶基因

write.csv(tf_region_df, paste0(output_path, "GRaNIE_tf_region_Links.csv"), row.names = FALSE)


region_gene_df <- results_df %>%
  dplyr::select(peak.ID, gene.name, gene.ENSEMBL, peak_gene.r, peak_gene.p_adj, peak_gene.distance) %>%
  dplyr::distinct() # 去重，因为一个 Peak-Gene 组合可能被多个 TF 结合

write.csv(region_gene_df, paste0(output_path, "GRaNIE_region_gene_Links.csv"), row.names = FALSE)



tf_gene_df <- results_df %>%
  dplyr::select(TF.name, gene.name, peak.ID, TF_peak.r, peak_gene.r)

write.csv(tf_gene_df, paste0(output_path, "GRaNIE_tf_gene_Links.csv"), row.names = FALSE)

