from jobseeker import JobSeeker,Lever, LeverApplier
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobtitle', type=str, help='The job title that you want to apply for',
                        default="machine learning engineer")
    parser.add_argument('--location',type=str,default="London")
    args = parser.parse_args()

    js = Lever()
    links = js.get_jobs(job_title=args.jobtitle, location=args.location)
    js.close_driver()
    for link in links:
        pass
    js = LeverApplier('https://jobs.lever.co/palantir/ff1029bd-bb6d-4d78-a03e-5f9744d0b798')
    js.fill_application()