import csv
from people_counter import PeopleCounter


def main():
    with open('data/input.txt', 'r') as file:
        label = [map(int, s[:-1].split(',')) for s in file.readlines()]

    with open('processed.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "id", "type", "enter", "exit", "door_y",
            "processed_enter", "processed_exit"])

    counter = PeopleCounter()
    for vid_id, vid_type, enter, exit, door_y in label:
        processed_enter, processed_exit = counter.process_video(
            door_y, f'data/{vid_id}.mp4', show=False, save=False)
        with open('processed.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
                "id", "type", "enter", "exit", "door_y",
                "processed_enter", "processed_exit"])
            writer.writerow({
                "id": vid_id,
                "type": vid_type,
                "enter": enter,
                "exit": exit,
                "door_y": door_y,
                "processed_enter": processed_enter,
                "processed_exit": processed_exit})
        print(f'Видео {vid_id} успешно обработано')


if __name__ == '__main__':
    main()
