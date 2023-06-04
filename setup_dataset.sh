# livedoorニュースコーパスのダウンロード
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar zxvf ldcc-20140209.tar.gz

# 整形結果格納用ファイル作成
echo -e "filename\tarticle"$(for category in $(basename -a `find ./text -type d` | grep -v text | sort); do echo -n "\t"; echo -n $category; done) > ./text/livedoor.tsv

# カテゴリごとに格納
for filename in `basename -a ./text/dokujo-tsushin/dokujo-tsushin-*`; do echo -n "$filename"; echo -ne "\t"; echo -n `sed -e '1,3d' ./text/dokujo-tsushin/$filename`; echo -e "\t1\t0\t0\t0\t0\t0\t0\t0\t0"; done >> ./text/livedoor.tsv
for filename in `basename -a ./text/it-life-hack/it-life-hack-*`; do echo -n "$filename"; echo -ne "\t"; echo -n `sed -e '1,3d' ./text/it-life-hack/$filename`; echo -e "\t0\t1\t0\t0\t0\t0\t0\t0\t0"; done >> ./text/livedoor.tsv
for filename in `basename -a ./text/kaden-channel/kaden-channel-*`; do echo -n "$filename"; echo -ne "\t"; echo -n `sed -e '1,3d' ./text/kaden-channel/$filename`; echo -e "\t0\t0\t1\t0\t0\t0\t0\t0\t0"; done >> ./text/livedoor.tsv
for filename in `basename -a ./text/livedoor-homme/livedoor-homme-*`; do echo -n "$filename"; echo -ne "\t"; echo -n `sed -e '1,3d' ./text/livedoor-homme/$filename`; echo -e "\t0\t0\t0\t1\t0\t0\t0\t0\t0"; done >> ./text/livedoor.tsv
for filename in `basename -a ./text/movie-enter/movie-enter-*`; do echo -n "$filename"; echo -ne "\t"; echo -n `sed -e '1,3d' ./text/movie-enter/$filename`; echo -e "\t0\t0\t0\t0\t1\t0\t0\t0\t0"; done >> ./text/livedoor.tsv
for filename in `basename -a ./text/peachy/peachy-*`; do echo -n "$filename"; echo -ne "\t"; echo -n `sed -e '1,3d' ./text/peachy/$filename`; echo -e "\t0\t0\t0\t0\t0\t1\t0\t0\t0"; done >> ./text/livedoor.tsv
for filename in `basename -a ./text/smax/smax-*`; do echo -n "$filename"; echo -ne "\t"; echo -n `sed -e '1,3d' ./text/smax/$filename`; echo -e "\t0\t0\t0\t0\t0\t0\t1\t0\t0"; done >> ./text/livedoor.tsv
for filename in `basename -a ./text/sports-watch/sports-watch-*`; do echo -n "$filename"; echo -ne "\t"; echo -n `sed -e '1,3d' ./text/sports-watch/$filename`; echo -e "\t0\t0\t0\t0\t0\t0\t0\t1\t0"; done >> ./text/livedoor.tsv
for filename in `basename -a ./text/topic-news/topic-news-*`; do echo -n "$filename"; echo -ne "\t"; echo -n `sed -e '1,3d' ./text/topic-news/$filename`; echo -e "\t0\t0\t0\t0\t0\t0\t0\t0\t1"; done >> ./text/livedoor.tsv

# 確認
head -10 ./text/livedoor.tsv