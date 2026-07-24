# Load Packages
library(Pando)
library(dplyr)
library(Seurat)
library(Matrix)
library(data.table)
library(BSgenome.Hsapiens.UCSC.hg38)
library(EnsDb.Hsapiens.v86)
library(Signac)
library(TFBSTools) # 处理 motifs 对象需要
library(GenomeInfoDb) # <--- 新增：必须加载，否则 keepStandardChromosomes 会报错

data(motifs)

cell_type <- "IMR90"

data_path <-paste0("/home/liyang/BioWuYan/dygmamba_project/data/cell_line/", cell_type, "/process/")

output_path <- paste0("/home/liyang/BioWuYan/dygmamba_project/data/cell_line/", cell_type, "/data_pando/")

dir.create(output_path, recursive = TRUE, showWarnings = FALSE)

atac_dir <- paste0(data_path,  "R/atac/")

rna_dir <- paste0(data_path,  "R/rna/")

message("正在准备基因组注释...")

# optional 1
# annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
# seqlevelsStyle(annotations) <- 'UCSC'
# genome(annotations) <- "hg38"
# annotations <- keepStandardChromosomes(annotations, pruning.mode = "coarse")

# optional 2
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)

seqlevels(annotations) <- paste0("chr", seqlevels(annotations))

genome(annotations) <- "hg38"
annotations <- keepStandardChromosomes(annotations, pruning.mode = "coarse")

################################################################################################
# ****************************************** Data Read

rna_mtx <- readMM(paste0(rna_dir, "sparse.mtx"))

rna_cellinfo <- fread(paste0(rna_dir, "cellinfo.csv"), header = T, data.table = F)

rna_info <- fread(paste0(rna_dir, "rnainfo.csv"), header = T, data.table = F)

gene_names <- make.unique(as.character(rna_info$Geneid))

rownames(rna_mtx) <- rna_cellinfo$V1
colnames(rna_mtx) <- gene_names

rna_meta <- rna_cellinfo
rownames(rna_meta) <- rna_cellinfo$V1

identical(rownames(rna_meta), rownames(rna_mtx))

rna_mtx_transposed <- t(rna_mtx)

rna_seurat <- CreateSeuratObject(count = rna_mtx_transposed, meta.data = rna_meta, assay = "RNA")

rna_seurat <- NormalizeData(rna_seurat, normalization.method = "LogNormalize", scale.factor = 10000)
rna_seurat <- FindVariableFeatures(rna_seurat, selection.method = "vst", nfeatures = 2000)

rna_seurat <- ScaleData(rna_seurat)


rna_seurat <- RunPCA(rna_seurat, features = VariableFeatures(object = rna_seurat))

ElbowPlot(rna_seurat)

rna_seurat <- RunUMAP(rna_seurat, dims = 1:30)

################################################################
# *************** ATAC-seq Data Read

atac_mtx <- readMM(paste0(atac_dir, "sparse.mtx"))

atac_cellinfo <- fread(paste0(atac_dir, "cellinfo.csv"), header = T, data.table = F)

atac_info <- fread(paste0(atac_dir, "atacinfo.csv"), header = T, data.table = F)

rownames(atac_mtx) <- atac_cellinfo$V1
colnames(atac_mtx) <- atac_info$V1

atac_meta <- atac_cellinfo
rownames(atac_meta) <- atac_cellinfo$V1

identical(rownames(atac_meta), rownames(atac_mtx))

# 行需为 feature
atac_mtx_transposed <- t(atac_mtx) 

atac_seurat <- CreateSeuratObject(count = atac_mtx_transposed, meta.data = atac_meta, assay = "ATAC")

all_peaks <- rownames(atac_seurat)

valid_pattern <- "^chr[a-zA-Z0-9_]+-[0-9]+-[0-9]+$"
is_valid_format <- grepl(valid_pattern, all_peaks)

standard_chroms <- seqlevels(annotations) 
peak_chroms <- sapply(strsplit(all_peaks, "-"), `[`, 1)
is_standard_chrom <- peak_chroms %in% standard_chroms

keep_peaks <- all_peaks[is_valid_format & is_standard_chrom]

message(paste0("原始 Peaks 数: ", length(all_peaks)))
message(paste0("清洗后 Peaks 数: ", length(keep_peaks)))

if (length(keep_peaks) == 0) stop("错误：过滤后没有剩余的 Peaks，请检查染色体命名格式或 annotations！")


atac_seurat <- subset(atac_seurat, features = keep_peaks)

atac_seurat <- RunTFIDF(atac_seurat)

atac_seurat <- FindTopFeatures(atac_seurat, min.cutoff = 'q0')

atac_seurat <- RunSVD(atac_seurat)

DepthCor(atac_seurat)

atac_seurat <- RunUMAP(atac_seurat, reduction = 'lsi', dims = 2:30)

################################################################

# 1. 找到两个对象共有的细胞, 先从 rna_obj 中筛选出这些共同细胞，创建你的新组合对象
#   从 atac_obj 中提取 ATAC assay, 同样筛选 ATAC assay 中的共同细胞
#   将筛选后的 ATAC assay 添加到筛选后的 RNA 对象中

common_cells <- intersect(colnames(rna_seurat), colnames(atac_seurat))

seurat_obj <- rna_seurat[, common_cells]

atac_assay <- GetAssay(atac_seurat, assay = "ATAC")

atac_assay_sub <- subset(atac_assay, cells = common_cells)

seurat_obj[["ATAC"]] <- atac_assay_sub

###################################

DefaultAssay(seurat_obj) <- "RNA"
seurat_obj <- RunPCA(seurat_obj) 

# ATAC 做 LSI (需要先做 RunTFIDF/RunSVD，参考上一个问题的回答)
DefaultAssay(seurat_obj) <- "ATAC"
seurat_obj <- RunSVD(seurat_obj)

# 2. 寻找多模态邻居 (WNN)
seurat_obj <- FindMultiModalNeighbors(
  seurat_obj, 
  reduction.list = list("pca", "lsi"), 
  dims.list = list(1:30, 2:30)
)

# 3. 基于 WNN 运行 UMAP 和 聚类
seurat_obj <- RunUMAP(seurat_obj, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
seurat_obj <- FindClusters(seurat_obj, graph.name = "wsnn", algorithm = 3, resolution = 0.5)

################################################################################################

counts <- GetAssayData(seurat_obj, assay = "ATAC", layer = "counts")

peaks_name <- rownames(counts)
peaks_gr <- StringToGRanges(peaks_name, sep = c("-", "-"))


# =======================================================
# 第二步：重新制造 ChromatinAssay
# =======================================================
message("正在重建 ChromatinAssay...")

# 读取本地下载好的 chromInfo 文件
chrom_info_df <- read.table("/home/liyang/BioWuYan/dygmamba_project/data/chrominfo/chromInfo.txt.gz", sep="\t", header=F)
# 构造 Seqinfo 对象
custom_seqinfo <- Seqinfo(
  seqnames = chrom_info_df$V1, 
  seqlengths = chrom_info_df$V2, 
  genome = "hg38"
)


# 这里的关键是把 annotation 放进去，解决那个 "UseMethod" 报错
chrom_assay <- CreateChromatinAssay(
  counts = counts,
  sep = c("-", "-"),
  ranges = peaks_gr,
  genome = custom_seqinfo,
  fragments = NULL,      # 如果你有 fragment 文件路径，填在这里；没有就填 NULL
  annotation = annotations
)

# =======================================================
# 第三步：替换旧的 Assay
# =======================================================
message("正在替换 Assay...")

seurat_obj[["ATAC"]] <- chrom_assay

DefaultAssay(seurat_obj) <- "RNA"

################################################################################################



################################################################################################
# ****************************************** GRN Inference

message("开始构建基因调控网络 (GRN)...")

grn_object <- initiate_grn(seurat_obj, rna_assay = 'RNA', peak_assay = 'ATAC')

grn_object <- find_motifs(
    grn_object, 
    pfm = motifs, 
    genome = BSgenome.Hsapiens.UCSC.hg38 #,
    # motif_tfs = motif_tfs_filtered # <--- 传入这个修正后的映射表
)

print("Motif 扫描完成！")


grn_object <- infer_grn(grn_object)

saveRDS(grn_object, file = paste0(output_path,"grn_seurat_object_tt.rds"))

print("*********************  save data ****************************")

################################################################################################
# ****************************************** GRN analysis


# 1. 提取所有推断出的调控参数
grn_df <- coef(grn_object)

# 2. 转换为标准的 DataFrame
grn_df <- as.data.frame(grn_df)

grn_significant <- subset(grn_df, padj < 0.05)

tf_gene_df <- grn_significant[, c('tf', 'target', 'estimate', 'padj')]
colnames(tf_gene_df) <- c('TF', 'Gene', 'Weight', 'padj')
tf_gene_df <- tf_gene_df %>% 
  distinct(TF, Gene, .keep_all = TRUE)

tf_region_df <- grn_significant[, c('tf', 'region', 'estimate', 'padj')]
colnames(tf_region_df) <- c('TF', 'Region', 'Weight', 'padj')
tf_region_df <- tf_region_df %>% 
  distinct(TF, Region, .keep_all = TRUE)

region_gene_df <- grn_significant[, c('region', 'target', 'estimate', 'padj')]
colnames(region_gene_df) <- c('Region', 'Gene', 'Weight', 'padj')
region_gene_df <- region_gene_df %>% 
  distinct(Region, Gene, .keep_all = TRUE)

write.csv(tf_gene_df, file = paste0(output_path, "tf_gene_network.csv"), row.names = FALSE)
write.csv(tf_region_df, file = paste0(output_path, "tf_region_network.csv"), row.names = FALSE)
write.csv(region_gene_df, file = paste0(output_path, "region_gene_network.csv"), row.names = FALSE)
