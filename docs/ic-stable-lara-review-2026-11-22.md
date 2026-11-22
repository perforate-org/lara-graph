# ic-stable-lara の設計の確認（2026-11-22）

lara-core で置き換える前提で、Gleaph の `crates/ic-stable-lara` の永続レイアウト、辺の置き方、Gleaph との境界を調べた。根拠は、Gleaph リポジトリのファイルと行番号で示す（パスは `crates/ic-stable-lara/src/` からの相対。`G/` は `crates/graph/src/`）。コードで確かめていない点は「未確認」と書く。

## 1. メモリの配置

- 1つの向き（orientation）につき、16 のメモリを使う（`GraphMemories`、labeled/graph/init.rs:35-68）。順方向と逆方向は、完全に独立した2つの `LabeledLaraGraph` である。保守キューと、向きごとの LTB を合わせて、Graph 側では約35の MemoryId を割り当てている（G/facade/stable/memory.rs:25-67）。
- 主な領域：

| 領域 | 中身 |
|---|---|
| vertices | 頂点の行（25B） |
| buckets | ラベル別バケットの記録（34B）。edge slab と同じ形式のストアを使う |
| bucket free spans | バケット記録用の空き span（FreeSpanStore と順序付きマップ） |
| edge_counts | PMA の木のノードごとの actual/total |
| edges | 辺の slab（4B/行） |
| edge_spill / values_spill | バケットごとの spill run（クラス 8〜1024 行） |
| edge_span_meta | 葉の span の位置 |
| inline_property_bytes_slab | inline プロパティのバイト列 |
| value_blobs | 大きなプロパティ値（StableBTreeMap） |
| ltb | tree のブロック（4KiB + 16B のヘッダ） |

- 再オープン時は、value_blobs と ltb 以外のすべての領域が「全部空」か「全部ある」かのどちらかでなければならない（init.rs:216-240）。

## 2. 頂点の行と、ラベル別バケットの記録

**頂点の行（`LabeledVertex`、25B。labeled/record.rs:1731-1770）**：`locator`（36ビットの開始位置 + 28ビットのメタ）、`degree`、`stored_slots`、inline プロパティの確保量（u40）、bypass 用の spill run ID。

- **通常モード**：開始位置は、バケット記録の列（buckets ストア内）を指す。`degree` は生きているバケットの数。
- **bypass モード**：既定のラベルだけを持ち、末尾の頂点 ID である頂点に限り、バケットを持たず、辺の slab を直接指す（labeled/graph/row_plane.rs:147-153）。別のラベルが来るか、末尾でなくなると、通常モードへ移る。

**ラベル別バケットの記録（`LabelBucket`、34B。record.rs:100-128、203）**：同じバイトが、モード（tiny / slab / tree）によって別の意味になる。

- ラベルのキーは `BucketLabelKey`（u16。15ビットのラベル + 有向かを表す最上位ビット。labeled/bucket_label_key.rs:25-115）。
- 1頂点のバケット記録は、buckets ストアの中で、ラベル順に並んだ連続した列になる。この列は、16頂点（1セグメント）単位で詰めて置かれる（labeled/bucket_store.rs:14-16、151-156）。
- **ラベルを追加するとき**：列に余白があれば後ろをずらす。なければ、**セグメント内の16頂点すべての列を、新しい span に書き直す**（bucket_store.rs:965-1029）。
- **ラベルを引くとき**：直前に引いたバケットのキャッシュと、64エントリのハッシュキャッシュを見たうえで、バケット数に応じて1〜2回の読み出し、二分探索（log2 d 回）、または列の一括読み出し（16個以上）を使う（labeled/graph/bucket.rs:306-500）。
- 1頂点あたりのバケット数は、形式上 65,536 個、運用上の上限が 1,024 個（labeled/graph.rs:97）。

## 3. 辺の置き方

| モード | 中身 | 移り変わり |
|---|---|---|
| tiny | descriptor の中に4本まで。新しいバケットはすべて tiny から始まる | 満杯で穴がなければ slab へ。**生きた辺だけをコピーするので、穴が詰まる**（insert.rs:1070-1089） |
| slab | PMA の葉（16頂点）の中の連続した prefix | 葉に余白がなければ spill へ |
| spill | バケットごとの run（8〜1024 行、クラスごとの空きリスト） | run が満杯になるか、領域が尽きたら tree へ |
| tree | LTB のブロック（1024 本/ブロック、最大深さ3） | 1024 本で昇格（tombstone ごとコピーし、位置を保つ）。512 本以下で降格（詰め直し、移動の通知を出す） |
| bypass | 既定のラベルだけの頂点が、slab と run を直接持つ | 上記のとおり |

- **PMA の葉**：16頂点、密度の閾値 1.0。重み付きの slide と relocation、FreeSpanStore（128 のサイズクラスと、順序付きマップによる隣の検索）。FreeSpanStore は、辺、バケット記録、値の3か所で使われている。
- **削除**：その場に tombstone（`VertexRef::tombstone()`）を書く。移動の通知は出さない。
- **再利用**：Unordered のラベルは、最初の穴を線形に探して再利用する（insert.rs:365-429）。Insertion のラベルは末尾に追加するだけ。
- **compaction**：保守処理の中でだけ行う。Unordered で run がなければ「最後の生きた辺を最初の穴へ移す」、それ以外は前に詰める（labeled/graph/compact.rs:6163-6287）。
- **位置（`BucketEntryPosition`）**：(頂点, ラベル) ごとに、tombstone を含めて数えた u32。移動の通知が出るのは、compaction、run の fold、run の作り直し、tree の降格。relocation、slide、tree への昇格では、位置は変わらない。

## 4. Gleaph との境界

- Graph が使うのは `DeferredBidirectionalLabeledLaraGraph` だけである（G/facade/stable/memory.rs:198）。
- **挿入**：有向（順方向に書いてから逆方向）と無向（両端点とも順方向のストアに書く）の2種類。書いた位置を返す。回復可能なエラーは、すべて最初の書き込みより前に返す。後半の書き込みで失敗したら trap する。
- **削除**：位置を指定して消し、移動の通知を受け取る。そのあと compaction を保守キューに積む。
- **走査**：ラベルと有向／無向で絞り込み、昇順か降順で読む。offset（生きた辺を飛ばす、降順）、位置からのカーソル（tombstone も limit に数える）、位置の集合を指定した読み出しがある。
- **点の読み出し**：LARA の点読み出しの API は使っていない。Graph 側がバケットを走査して位置を比べている（G/facade/store/lookup.rs:13-31）。
- **移動の通知**：Graph は順方向の移動だけを受け取り、sidecar のプロパティを付け替え、索引を「古い位置から削除、新しい位置に追加」で出し直す（G/facade/store/sidecar.rs:35-133）。
- **逆向きの辺の対応付け**：索引は持たず、pair ordinal（同じ終点を持つ生きた辺の順位）で毎回数え直す（labeled/bidirectional/counterpart.rs:435-507）。
- **保守処理**：8種類の作業を、stable memory 上のキューで管理する。IC の命令数の上限のため、挿入直後の処理予算は wasm では 0 にしてある。タイマーで 1秒（予算切れ）か 5秒ごとに進める（G/facade/ic_budget.rs、G/facade/maintenance_timer.rs）。
- **inline プロパティ**：辺とは別のバイトのストアに置く。幅はバケットごとの u16。幅を 0 から w に変えるときは段階的な処理になり、挿入は「保留中」を返して再試行を求める。
- **頂点の削除**：段階的に行う。各段階で辺を1本ずつ消し、逆向きの対応する辺も消し、通知を出す。

## 5. 調査で見つかった、不具合の可能性がある箇所（未確認）

1. **pair ordinal の整合**：Unordered の「最初の穴を再利用」と「最後の辺を最初の穴へ移す compaction」には、同じ終点を持つ辺の順序を確かめる処理が見当たらなかった。多重辺で、逆向きの辺との対応がずれる可能性がある（ADR 0048 §4-5 の要件）。
2. **通知なしで位置が変わる経路**：tiny から slab への昇格で穴を詰める処理（insert.rs:1070-1089）と、bypass の行を前に詰める処理（compact.rs:6842-6912）に、`EdgeSlotMove` が見当たらなかった。
3. **文書とコードの食い違い**：record.rs のコメント（29B、ビット60/63、tiny の容量）、bucket_store.rs（セグメントは32頂点）、lara.md（葉ごとの log）、ADR 0096/0097 の表、MemoryId の在庫表などが古いまま。

## 6. lara-core が保つべきものと、物理設計のためだけにあるもの

**保つべきもの**

- 有向と無向の、対になった挿入と削除。1メッセージの中で原子的に行い、書いた位置を返すこと。
- (頂点, ラベル, 位置) のハンドル。位置が変わるときは移動の通知を出すこと。
- 多重辺を含む、逆向きの辺の対応付けと、正本のハンドルの解決。
- ラベルと有向／無向による絞り込み、昇順と降順、offset、位置からのカーソル。
- 固定幅のラベルの inline プロパティの一括読み出し。
- ラベルごとの挿入順（Insertion）。
- 段階的な頂点の削除と、辺ごと・完了時の通知。
- 予算付きで再開できるバックグラウンド処理。

**物理設計のためだけにあるもの**

- slab / spill / tree / tiny / bypass のモード、PMA の密度と slide、FreeSpanStore と順序付きマップ。
- 8種類の保守作業の大半、プロパティの段階的な実体化と再試行、配置情報の問い合わせ、片方の向きだけを直す修復用の書き込み。
- Graph 側でバケットを走査する点の読み出し。

## 7. lara-core の設計への影響

- **位置の扱いは、これまでの方針と同じでよい。** tombstone を含めた位置、削除では位置を変えない、位置を変えるのは compaction だけで、そのときに通知を出す。
- **ラベル別バケットを別メモリに置く案は、ic-stable-lara と同じ方向である。** ただし ic-stable-lara は、記録の列を16頂点単位で詰めているため、ラベルの追加でセグメント全体を書き直す。lara-core は、列をページストアのチャンクに置くので、頂点ごとに独立して伸ばせる。
- **bypass は、「ラベル1つの頂点を埋め込む」最適化を、既定のラベルに限って行ったもの**である。条件（末尾の頂点のみ）とモードの切り替えが複雑さの原因になっている。
- **有向かどうかは、ラベルのキーの1ビット**で表し、無向の辺は両端点とも順方向のストアに置く。lara-core もこの形に合わせる。
- **保守処理を全部なくせるかは、Graph 側の移動のコスト次第である。** lara-core の compaction は1回あたり1024件以下の移動に収まるが、Graph は移動1件ごとに sidecar の付け替えと索引の出し直しを行う。このコストが命令数の上限に効くなら、compaction だけは予算付きで後回しにする仕組みが要る。
- **inline プロパティは大きな部品である。** 辺と並べて置く位置の対応（ADR 0099 を含む）は、別途詳しく調べる必要がある。
