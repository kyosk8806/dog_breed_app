from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={'root_dir': 'Yorkshire_terrier'})
crawler.crawl(keyword="ヨークシャーテリア", max_num=600)
