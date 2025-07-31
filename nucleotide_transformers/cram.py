import pysam

# Path to your downloaded CRAM file
cram_path = "/home/ibab/HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam.cram"

# Open the CRAM file
samfile = pysam.AlignmentFile(cram_path, "rc", reference_filename="/home/ibab/hs37d5.fa")  # 'rc' = read CRAM

# Print details of the first alignment
for read in samfile.fetch():
    print("Read Name:", read.query_name)
    print("Reference Start:", read.reference_start)
    print("Mapping Quality:", read.mapping_quality)
    print("Base Qualities:", read.query_qualities)
    print("Flags:", read.flag)
    print("Sequence:", read.query_sequence)
    break


# import pysam
#
# samfile = pysam.AlignmentFile(
#     "/home/ibab/HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam.cram",
#     "rc",  # r = read, c = cram
#     reference_filename="/home/ibab/hs37d5.fa"
# )
#
# # Try printing first read
# for read in samfile.fetch():
#     print(read)
#     break
