import requests
import csv
import sys
import time

from bs4 import BeautifulSoup

# categories in sloganlist.com website as of 15 Feb 2020
category_list = [
    "drinking",
    "food",
    "restaurant",
    "car",
    "apparel",
    "technology",
    "business",
    "company",
    "cosmetic",
    "household",
    "tours",
    "airlines",
    "financial",
    "health-medicine",
    "education",
]
# name of the output file
output_file = "slogan_logo_list.csv"


def get_data(url):
    # In case of Status 403 (Forbidden), wait for some time (maybe hours) before retrying
    headers = {
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36",
        "Sec-Fetch-Dest": "document",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        return requests.get(url, headers=headers).text
    except Exception:
        time.sleep(1)
        return requests.get(url, headers=headers).text
    print(requests.get(url).status_code)


def collect_data(category_list=None):
    rows = []
    print(category_list)
    # iterate over each category
    for category in category_list:
        print("Category:" + category)
        # generate base url
        base_url = "https://www.sloganlist.com/" + str(category) + "-slogans/"
        page = 1
        # iterate over all of the pages
        while True:
            print("Page Number:" + str(page))
            # generate page-specific url
            url = base_url
            if page > 1:
                url = base_url + "index_" + str(page) + ".html"

            # fetch data and parse it with bs4
            data = get_data(url)
            soup = BeautifulSoup(data, "html.parser")

            # get org names and logos
            org_names_logos = soup.findAll("div", {"class": "list-group-item-heading"})
            # get org slogans
            org_slogans = soup.findAll("p", {"class": "list-group-item-text"})

            # break out of while loop if no more companies
            if len(org_names_logos) == 0 and len(org_slogans) == 0:
                break

            for i in range(0, len(org_slogans)):
                # try to extract name, logo, and slogan
                try:
                    if (
                        org_names_logos[i].contents[0]["src"]
                        == "/theme/default/images/nopic.jpg"
                    ):
                        continue
                    row = [
                        org_names_logos[i].contents[1].strip(),
                        org_slogans[i].contents[0].strip(),
                        "https://www.sloganlist.com"
                        + org_names_logos[i].contents[0]["src"],
                    ]
                    rows.append(row)
                    # print(row)
                except (IndexError, AttributeError):
                    pass
            page += 1
    return rows


def write_data(data, output_file):
    # write data to the output file
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        # header of the csv
        writer.writerow(["Company", "Slogan"])
        # contents
        writer.writerows(data)
    pass


def main():
    data = collect_data(category_list)
    write_data(data, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())